from datetime import datetime
import json
import logging
import os
from uuid import UUID, uuid4
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from app.configs.settings import settings
from app.utils.examples import get_summary_examples, get_entity_extraction_examples, get_preference_extraction_examples
from app.utils.llm import llm_factory, estimate_tokens, prepare_conversation_data
from app.models.agents import AgentState, SmileMessage, ExtractorType
from app.models.memory import(
     EntityExtractorResponse, 
     PersonEntity, 
     OrganizationEntity, 
     Relationship, 
     ConversationSummary,
     PreferenceExtractorResponse,
     Preference
)
from app.services.neo4j import(
     create_entity_node, 
     create_entity_relationship,
     create_summary_node, 
     driver,
     create_preference_node,
     fetch_existing_preference_types,
     get_person_id_by_name
)
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from langchain_core.messages.modifier import RemoveMessage
import uuid

logger = logging.getLogger(__name__)

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = settings.app_config["langchain_config"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"] = settings.app_config["langchain_config"]["project"]


class SmileMemory:
    def __init__(self):
        self.settings = settings
        self.logger = logger

        self.checkpoint_path = settings.app_config.get("checkpoint_path")
        self.current_state = None
        self.graph = None
      

    def initialise_entity_extractor(self):
        self.entity_extractor_llm = llm_factory(self.settings, "entity_extractor_agent")
        self.entity_extractor_prompt = PromptTemplate.from_template(self.settings.llm_config["entity_extractor_agent"]["prompt_template"])
        self.entity_extractor_llm = self.entity_extractor_llm.with_structured_output(EntityExtractorResponse)
        self.entity_extractor_chain = self.entity_extractor_prompt | self.entity_extractor_llm

    def initialise_preference_extractor(self):
        self.preference_extractor_llm = llm_factory(self.settings, "preference_extractor_agent")
        self.preference_extractor_prompt = PromptTemplate.from_template(self.settings.llm_config["preference_extractor_agent"]["prompt_template"])
        self.preference_extractor_llm = self.preference_extractor_llm.with_structured_output(PreferenceExtractorResponse)
        self.preference_extractor_chain = self.preference_extractor_prompt | self.preference_extractor_llm

    def initialise_conversation_summarizer(self):
        self.conversation_summarizer_llm = llm_factory(self.settings, "conversation_summarizer_agent")
        self.conversation_summarizer_prompt = PromptTemplate.from_template(self.settings.llm_config["conversation_summarizer_agent"]["prompt_template"])
        self.conversation_summarizer_llm = self.conversation_summarizer_llm.with_structured_output(ConversationSummary)
        self.conversation_summarizer_chain = self.conversation_summarizer_prompt | self.conversation_summarizer_llm



    def process_entity_extraction_batch(self, messages: List[BaseMessage]):
        """
        Process a batch of SmileMessage instances, extract entities and relationships,
        and store them in the Neo4j database.

        Args:
            messages (List[SmileMessage]): List of SmileMessage instances to process.
            state (AgentState): The agent's current state.

        Logging:
            - Info when processing messages.
            - Error when exceptions occur.
            - Debug for detailed internal states.
        """
        max_retries = 3
        retry_count = 0
        examples = get_entity_extraction_examples()

        while retry_count < max_retries:
            try:
                # Convert messages to conversation text
                conversation_text = prepare_conversation_data(messages)

                # Create a dictionary with the required variables
                prompt_variables = {
                    "conversation_text": conversation_text,
                    "examples": examples
                }

                # Invoke the chain with the variables dictionary
                self.logger.debug(f"Invoking LLM with prompt variables: {prompt_variables}")
                response = self.entity_extractor_chain.invoke(prompt_variables)
                break

            except ValidationError as ve:
                retry_count += 1
                self.logger.error(f"ValidationError on attempt {retry_count}: {ve}")
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached. Unable to process message batch.")
                    raise
                continue
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred: {e}")
                raise

        # Proceed with processing the response
        entity_map = {}

        # Process persons
        for person_resp in response.persons:
            person = PersonEntity(
                id=uuid4(),
                name=person_resp.name,
                type=person_resp.type,
                category=person_resp.category,
                nickname=person_resp.nickname,
                birth_date=person_resp.birth_date,
                email=person_resp.email,
                phone=person_resp.phone,
                address=person_resp.address,
                notes=person_resp.notes,
                created_at=datetime.utcnow(),
                updated_at=None
            )
            entity_map[person.name] = person

        # Process organizations
        for org_resp in response.organizations:
            organization = OrganizationEntity(
                id=uuid4(),
                name=org_resp.name,
                type=org_resp.type,
                industry=org_resp.industry,
                website=org_resp.website,
                address=org_resp.address,
                notes=org_resp.notes,
                created_at=datetime.utcnow(),
                updated_at=None
            )
            entity_map[organization.name] = organization

        # Process relationships
        relationships = []
        for rel_resp in response.relationships:
            from_entity = entity_map.get(rel_resp.from_entity_name)
            to_entity = entity_map.get(rel_resp.to_entity_name)
            if from_entity and to_entity:
                relationship = Relationship(
                    id=uuid4(),
                    from_entity_id=from_entity.id,
                    to_entity_id=to_entity.id,
                    type=rel_resp.type,
                    since=rel_resp.since,
                    until=rel_resp.until,
                    notes=rel_resp.notes
                )
                relationships.append(relationship)
            else:
                self.logger.warning(
                    f"Entity not found for relationship {rel_resp.type} between {rel_resp.from_entity_name} and {rel_resp.to_entity_name}"
                )

        # Now, write entities and relationships to Neo4j
        try:
            with driver.session() as session:
                # Create entities and update entity_map with database IDs
                for name, entity in entity_map.items():
                    db_id = session.execute_write(create_entity_node, entity)
                    entity.db_id = db_id  # Store the database ID
                    entity_map[name] = entity  # Update the entity_map

                # Build a mapping from IDs to entities
                id_to_entity_map = {entity.id: entity for entity in entity_map.values()}

                # Update relationships to use database IDs
                for rel in relationships:
                    from_entity = id_to_entity_map.get(rel.from_entity_id)
                    to_entity = id_to_entity_map.get(rel.to_entity_id)
                    if from_entity and to_entity:
                        # Assign the database IDs to the relationship
                        rel.from_entity_db_id = from_entity.db_id
                        rel.to_entity_db_id = to_entity.db_id

                        # Debug logging
                        self.logger.debug(
                            f"Creating relationship {rel.type} between DB IDs {rel.from_entity_db_id} and {rel.to_entity_db_id}"
                        )

                        session.execute_write(create_entity_relationship, rel)
                    else:
                        self.logger.warning(
                            f"Entity not found for relationship {rel.type} between IDs {rel.from_entity_id} and {rel.to_entity_id}"
                        )

        except Exception as e:
            self.logger.error(f"An error occurred while writing to Neo4j: {e}")
            raise


        return response


        

    def execute_graph(self, state: AgentState, config: dict):

        self.initialise_entity_extractor()
        self.initialise_preference_extractor()
        self.initialise_conversation_summarizer()

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("extractor", self.extractor)
        graph_builder.set_entry_point("extractor")
        graph_builder.add_edge("extractor", END)
        
        with SqliteSaver.from_conn_string(conn_string=self.checkpoint_path) as checkpointer:
            self.graph = graph_builder.compile(checkpointer=checkpointer)   

        
            response = self.graph.invoke(state, config=config)
        
        return response

    def process_preference_extraction_batch(self, batch: List[BaseMessage]) -> List[Preference]:
        """
        Process a batch of messages to extract preferences.

        Args:
            batch (List[BaseMessage]): The batch of messages to process.

        Returns:
            List[Preference]: The list of extracted preferences.
        """
        # Convert messages to conversation text
        conversation_text = "\n".join([msg.content for msg in batch])

        # Retrieve existing preference types from Neo4j
        with driver.session() as session:
            existing_types = session.execute_read(fetch_existing_preference_types)
        examples = get_preference_extraction_examples()
        # Prepare the prompt, including existing preference types
        prompt_values = {
            "conversation_text": conversation_text,
            "existing_preference_types": ", ".join(existing_types),
            "examples": examples
        }

        # Invoke the LLM with the prompt
        self.logger.debug(f"Invoking Preference Extractor LLM with prompt: {prompt_values}")
        response = self.preference_extractor_chain.invoke(prompt_values)

        # Parse the response into Preference models
        preferences = []
        for pref_resp in response.preferences:
            # Find or create person_id
            person_name = pref_resp.person_name
            with driver.session() as session:
                person_id = session.execute_read(get_person_id_by_name, person_name)
            if person_id is None:
                logger.error(f"Could not find person with name: {person_name}")
                return None  # or handle this case as needed
            
            try:
                # Skip preferences with empty details
                if not pref_resp.details:
                    logger.debug(f"Skipping preference for {person_name} - empty details")
                    continue
                preference = Preference(
                    person_id=uuid.UUID(person_id),  # Convert string to UUID
                    preference_type=pref_resp.preference_type,
                    importance=pref_resp.importance,
                    details=pref_resp.details,
                    created_at=datetime.now(),
                    updated_at=datetime.now()  # Add the required updated_at field
                )
                preferences.append(preference)
            except ValueError as e:
                logger.error(f"Error creating preference: {e}")
                return None
        
        # Store preferences in Neo4j
        with driver.session() as session:
            for preference in preferences:
                session.execute_write(create_preference_node, preference)


        # Return the list of extracted preferences
        return preferences

    def summarize_and_replace_batch(self, batch: List[BaseMessage]):
        """
        Summarize the batch of messages, remove original messages using RemoveMessage,
        and add the summary message.

        Args:
            batch (List[BaseMessage]): The batch of messages to summarize.
            state (AgentState): The current agent state.

        Returns:
            BaseMessage: The summary message generated.
        """
        # Prepare the conversation data for the batch
        conversation_data = prepare_conversation_data(batch)
        conversation_text = json.dumps(conversation_data, indent=2)
        examples = get_summary_examples()

        # Create the summarization prompt
        prompt_variables ={
            "conversation_text": conversation_text,
            "examples": examples
        }

        # Invoke the summarizer LLM
        summary_response = self.conversation_summarizer_chain.invoke(prompt_variables)

        # Create a summary message
        conversation_summary = ConversationSummary(**summary_response.model_dump())
        

        summary_message = self.create_summary_message(conversation_summary)


        for msg in batch:
            RemoveMessage(id=msg.id)
        

        self.logger.info(f"Replaced batch of {len(batch)} messages with summary message.")

        # Return the summary message
        return summary_message

    def extractor(self, state: AgentState):
        """
        Process messages in batches using extractors and summarizer.

        Args:
            state (AgentState): The current agent state containing messages.

        Returns:
            AgentState: Updated state after processing messages.
        """
        # Define your batch size
        batch_size = 50

        # Get the list of messages
        messages = state.messages


        # Get the total number of messages to process
        total_messages = len(messages)

        # If the message count is lower than the batch size, exit the flow
        if total_messages < batch_size:
            self.logger.info("Not enough messages to process a batch.")
            return state

        # Initialize lists to collect outputs
        all_entities = []
        all_summaries = []

        # Process messages in batches
        num_batches = total_messages // batch_size
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = start_index + batch_size
            batch = messages[start_index:end_index]

            # Process the batch through extractors
            entities_response = self.process_entity_extraction_batch(batch)
            all_entities.append(entities_response)
            self.logger.debug(f"Extracted entities: {entities_response}")
            # Process preference extraction
            preferences = self.process_preference_extraction_batch(batch)
            self.logger.debug(f"Extracted preferences: {preferences}")

            

            # Summarize the batch and replace messages
            summary_message = self.summarize_and_replace_batch(batch)

            all_summaries.append(summary_message)
            self.logger.debug(f"Generated summary message: {summary_message.content}")

        state.messages.extend(all_summaries)

        return state

    
    

    def create_summary_message(self, conversation_summary):
        """
        Creates a summary message excluding message_ids from the content
        
        Args:
            conversation_summary: The complete conversation summary object
            
        Returns:
            HumanMessage: Message containing filtered summary content
        """
        logger.debug("Creating summary message from conversation summary")
        
        # Convert to dictionary and remove message_ids
        summary_dict = conversation_summary.model_dump(mode='json')
        summary_dict.pop('message_ids', None)
        
        # Convert filtered dict back to JSON string
        filtered_content = json.dumps(summary_dict)
        
        logger.debug(f"Created filtered summary content: {filtered_content}")
        return HumanMessage(content=filtered_content)



if __name__ == "__main__":
    # Initialize the memory agent
    memory_agent = SmileMemory()

    # Create a test AgentState with some messages
    from langchain_core.messages import HumanMessage

    message=HumanMessage(content="Hello, how are you?")
    state = {"messages": []}
    
    message.pretty_print()


    config = {"configurable": {"thread_id": "123",
                               "checkpoint_id": "1efa61d0-53e9-67df-81d2-1efddde93b3c"
                               }}
    response = memory_agent.execute_graph(state=state, config=config)


