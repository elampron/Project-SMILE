# app/agents/memory.py
import asyncio
from datetime import datetime
import json
import logging
import os
from uuid import UUID, uuid4
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.configs.settings import settings
from app.utils.examples import get_summary_examples, get_entity_extraction_examples, get_preference_extraction_examples,get_cognitive_memory_examples
from app.utils.llm import llm_factory, prepare_conversation_data
from app.models.agents import AgentState, ExtractorType
from app.services.embeddings import EmbeddingsService
from app.models.memory import(
    EntityExtractorResponse, 
    PersonEntity, 
    OrganizationEntity, 
    Relationship, 
    ConversationSummary,
    PreferenceExtractorResponse,
    Preference,
    CognitiveMemory,
    MemoryAssociation,
    ValidationStatus,
    MemoryIndex,
    SemanticAttributes,
    TemporalContext,
    MemoryRelations
)
from app.services.neo4j import(
    create_entity_node, 
    create_entity_relationship,
    create_summary_node, 
    driver,
    create_preference_node,
    fetch_existing_preference_types,
    get_person_id_by_name,
    create_summary_relationships,
    create_cognitive_memory_node,
    initialize_schema_with_session
)
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from langchain_core.messages.modifier import RemoveMessage
import uuid
import tiktoken
from app.services.neo4j.entities import get_or_create_person_entity, create_entity_node
from app.services.neo4j.relationships import create_entity_relationship
from neo4j import ManagedTransaction


from app.utils.logger import logger

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = settings.app_config["langchain_config"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"] = settings.app_config["langchain_config"]["project"]


class SmileMemory:
    def __init__(self):
        self.settings = settings
        self.logger = logger

        self.postgres_url = settings.app_config.get("postgres_config")["conn"]
        self._checkpointer = None
        self._initialized = False
        self._saver = None
        self.graph = None
        self.embeddings_service = EmbeddingsService()
        
        # Schema initialization is now handled by API startup
        # No need to initialize here

    def initialize(self):
        """Initialize the memory system."""
        try:
            if not self._initialized:
                try:
                    self.logger.info(f"Connecting to PostgreSQL at {self.postgres_url}...")
                    
                    # Create direct connection with optimized settings
                    from psycopg import Connection
                    conn = Connection.connect(
                        self.postgres_url,
                        autocommit=True,
                        prepare_threshold=None,  # Disable prepared statements
                        options="-c synchronous_commit=off"  # Optimize for performance
                    )
                    
                    # Create saver with the connection and use it directly as checkpointer
                    self._checkpointer = PostgresSaver(conn)
                    
                    # Setup tables using the checkpointer
                    self._checkpointer.setup()
                    self.logger.info("PostgreSQL tables created successfully")
                    self.logger.info("Successfully connected to PostgreSQL")
                except Exception as pg_error:
                    self.logger.error(
                        "Failed to connect to PostgreSQL. Please ensure:\n"
                        "1. PostgreSQL is running\n"
                        "2. The connection string in app_config.yaml is correct\n"
                        "3. If running locally, use 'localhost' instead of 'postgres' as the host\n"
                        f"Error details: {str(pg_error)}"
                    )
                    raise
                
                if self.graph is None:
                    self._initialize_graph()
                self._initialized = True
            
            return self
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            if hasattr(self, '_checkpointer') and hasattr(self._checkpointer, 'conn'):
                try:
                    self._checkpointer.conn.close()
                except Exception:
                    pass
            raise

    def cleanup(self):
        """Cleanup PostgreSQL resources."""
        if self._saver and self._initialized:
            try:
                self._saver.__exit__(None, None, None)
                self._initialized = False
                self._checkpointer = None
                self._saver = None
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
                raise

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

    def initialise_cognitive_memory_extractor(self):
        """Initialize the cognitive memory extractor chain."""
        self.cognitive_memory_extractor_llm = llm_factory(self.settings, "cognitive_memory_extractor_agent")
        self.cognitive_memory_extractor_prompt = PromptTemplate.from_template(self.settings.llm_config["cognitive_memory_extractor_agent"]["prompt_template"])
        self.cognitive_memory_extractor_llm = self.cognitive_memory_extractor_llm.with_structured_output(CognitiveMemory)
        self.cognitive_memory_extractor_chain = self.cognitive_memory_extractor_prompt | self.cognitive_memory_extractor_llm

    def _extract_entities(self, messages: List[BaseMessage]) -> Optional[EntityExtractorResponse]:
        """
        Extract entities from a batch of messages.
        
        Args:
            messages (List[BaseMessage]): Messages to process
            
        Returns:
            Optional[EntityExtractorResponse]: Extracted entities or None if extraction fails
            
        Raises:
            ValidationError: If response validation fails
            Exception: For other unexpected errors
            
        Logging:
            - Debug logs for LLM invocation
            - Error logs for validation and unexpected errors
        """
        max_retries = 3
        retry_count = 0
        examples = get_entity_extraction_examples()

        while retry_count < max_retries:
            try:
                # Convert messages to conversation text
                conversation_text = prepare_conversation_data(messages)
                
                # Create prompt variables
                prompt_variables = {
                    "conversation_text": json.dumps(conversation_text, indent=2),
                    "examples": examples
                }
                
                # Invoke the chain with the variables dictionary
                self.logger.debug(f"Invoking Entity Extractor LLM with prompt variables: {prompt_variables}")
                response = self.entity_extractor_chain.invoke(prompt_variables)
                return response

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

    def _initialize_graph(self):
        """Initialize the agent graph with tools and checkpointer."""
        self.logger.info("Initializing agent graph")
        
        
          # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("extractor", self.extractor)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("extractor")
        workflow.add_edge("extractor", END)

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.graph=workflow.compile(
            checkpointer=self._checkpointer,
            interrupt_before=None,
            interrupt_after=None,
            debug=False,
        )

    def process_entity_extraction_batch(self, messages: List[BaseMessage]):
        """Process a batch of messages for entity extraction."""
        try:
            # Get entity extraction response
            response = self._extract_entities(messages)
            if not response:
                return
            
            # Process in a Neo4j transaction
            with driver.session() as session:
                def process_entities(tx):
                    entity_map = {}
                    
                    # Process persons using centralized function
                    for person_resp in response.persons:
                        person_details = {
                            'name': person_resp.name,
                            'type': person_resp.type,
                            'category': person_resp.category,
                            'nickname': person_resp.nickname,
                            'birth_date': person_resp.birth_date,
                            'email': person_resp.email,
                            'phone': person_resp.phone,
                            'address': person_resp.address,
                            'notes': person_resp.notes
                        }
                        person = get_or_create_person_entity(tx, person_details)
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
                        create_entity_node(tx, organization)
                        entity_map[organization.name] = organization
                    
                    # Process relationships
                    for rel in response.relationships:
                        from_entity = entity_map.get(rel.from_entity_name)
                        to_entity = entity_map.get(rel.to_entity_name)
                        
                        if from_entity and to_entity:
                            relationship = Relationship(
                                from_entity_id=from_entity.id,
                                to_entity_id=to_entity.id,
                                from_entity_db_id=from_entity.db_id,  # Set the db_id from the entity
                                to_entity_db_id=to_entity.db_id,  # Set the db_id from the entity
                                type=rel.type,
                                since=rel.since,
                                until=rel.until,
                                notes=rel.notes
                            )
                            create_entity_relationship(tx, relationship)
                
                # Execute transaction
                session.execute_write(process_entities)
                
        except Exception as e:
            self.logger.error(f"Error processing entity extraction batch: {str(e)}")
            raise

    def execute_graph(self, state: AgentState, config: dict):
        """Execute the memory graph with async support."""
        # Initialize if not already initialized
        if not self._initialized:
            self.initialize()
            
        self.initialise_entity_extractor()
        self.initialise_preference_extractor()
        self.initialise_conversation_summarizer()
        self.initialise_cognitive_memory_extractor()

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("extractor", self.extractor)
        graph_builder.set_entry_point("extractor")
        graph_builder.add_edge("extractor", END)
        
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

    def summarize_and_replace_batch(self, batch: List[BaseMessage]) -> ConversationSummary:
        """Summarize the batch of messages and handle timestamps separately."""
        
        # Get conversation data and create prompt
        conversation_data = prepare_conversation_data(batch)
        examples = get_summary_examples()
        prompt_variables = {
            "conversation_text": json.dumps(conversation_data, indent=2),
            "examples": examples
        }

        # Get summary from LLM
        summary_response = self.conversation_summarizer_chain.invoke(prompt_variables)
        
        # Create summary with timestamps from messages
        try:
            # Find earliest and latest message timestamps
            timestamps = [msg.additional_kwargs.get('timestamp') for msg in batch if msg.additional_kwargs.get('timestamp')]
            start_time = min(timestamps) if timestamps else datetime.utcnow()
            end_time = max(timestamps) if timestamps else datetime.utcnow()
            
            # Convert response to dict and remove time fields that we'll set manually
            summary_dict = summary_response.model_dump()
            summary_dict.pop('start_time', None)
            summary_dict.pop('end_time', None)
            
            conversation_summary = ConversationSummary(
                **summary_dict,
                start_time=start_time,
                end_time=end_time
            )
        except ValidationError as e:
            self.logger.error(f"Validation error creating ConversationSummary: {e}")
            raise

        # Handle message deletion and storage
        deleted_messages = [RemoveMessage(id=msg.id) for msg in batch]
        
        try:
            with driver.session() as session:
                session.execute_write(create_summary_node, conversation_summary)
                session.execute_write(create_summary_relationships, conversation_summary)
            logger.info(f"Stored summary in Neo4j: {conversation_summary.id}")
        except Exception as e:
            logger.error(f"Error storing summary in Neo4j: {str(e)}")
            raise

        return conversation_summary, deleted_messages

    def count_tokens_in_messages(self, messages: List[BaseMessage]) -> int:
        """
        Count the total number of tokens in a list of messages.
        
        Args:
            messages (List[BaseMessage]): List of messages to count tokens for
            
        Returns:
            int: Total number of tokens
            
        Logging:
            - Debug logs for token counts
        """
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        total_tokens = 0
        
        for msg in messages:
            tokens = len(enc.encode(msg.content))
            total_tokens += tokens
            self.logger.debug(f"Message tokens: {tokens}")
        
        return total_tokens

    def create_token_based_batch(
        self, 
        messages: List[BaseMessage], 
        max_tokens: int = 5000,
        max_messages: int = 25
    ) -> Tuple[List[BaseMessage], List[BaseMessage]]:
        """
        Create a batch of messages that fits within both token and message count limits.
        
        Args:
            messages (List[BaseMessage]): Messages to batch
            max_tokens (int): Maximum tokens per batch
            max_messages (int): Maximum number of messages per batch
            
        Returns:
            Tuple[List[BaseMessage], List[BaseMessage]]: (current_batch, remaining_messages)
            
        Logging:
            - Info for batch creation
            - Debug for token counts and message counts
            - Warning for large messages that need splitting
        """
        current_batch = []
        current_tokens = 0
        
        for i, msg in enumerate(messages):
            # Check if we've hit the max messages limit
            if len(current_batch) >= max_messages:
                self.logger.info(f"Reached max messages limit ({max_messages})")
                return current_batch, messages[i:]
            
            msg_tokens = self.count_tokens_in_messages([msg])
            
            # Handle messages that exceed max_tokens
            if msg_tokens > max_tokens:
                self.logger.warning(f"Message exceeds token limit ({msg_tokens} > {max_tokens}). Splitting message.")
                
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                tokens = enc.encode(msg.content)
                tokens_to_keep = max_tokens - 100  # Leave buffer for safety
                truncated_tokens = tokens[:tokens_to_keep]
                truncated_content = enc.decode(truncated_tokens)
                truncated_message = HumanMessage(content=truncated_content)
                
                # Add to current batch if we have space and haven't hit message limit
                if current_tokens < max_tokens and len(current_batch) < max_messages:
                    current_batch.append(truncated_message)
                    current_tokens += len(truncated_tokens)
                    return current_batch, messages[i+1:]
                else:
                    return current_batch, [truncated_message] + messages[i+1:]
                continue
                
            # Normal case - check both token and message limits
            if (current_tokens + msg_tokens > max_tokens or 
                len(current_batch) >= max_messages) and current_batch:
                self.logger.info(f"Created batch with {current_tokens} tokens and {len(current_batch)} messages")
                return current_batch, messages[i:]
            
            current_batch.append(msg)
            current_tokens += msg_tokens
        
        self.logger.info(f"Created final batch with {current_tokens} tokens and {len(current_batch)} messages")
        return current_batch, []

    def extractor(self, state: AgentState):
        """
        Process messages in token-based batches using extractors and summarizer.
        
        Args:
            state (AgentState): The current agent state containing messages
            
        Returns:
            AgentState: Updated state after processing messages
            
        Logging:
            - Info for batch processing
            - Debug for extraction results
        """
        MAX_TOKENS = 50000
        messages = state.messages
        remaining_messages = messages
        all_entities = []
        all_memories = []
        all_summaries = []
        
        self.logger.info(f"Starting to process {len(messages)} messages")
        
        while remaining_messages:
            # Create batch based on token count
            current_batch, remaining_messages = self.create_token_based_batch(
                remaining_messages, 
                MAX_TOKENS
            )
            
            if not current_batch:
                break
            
            self.logger.info(f"Processing batch of {len(current_batch)} messages")
            
            # Process the batch through extractors
            entities_response = self.process_entity_extraction_batch(current_batch)
            all_entities.append(entities_response)
            self.logger.debug(f"Extracted entities: {entities_response}")

            # First: Process cognitive memories
            memories = self.process_cognitive_memory_batch(current_batch)
            all_memories.extend(memories)
            self.logger.debug(f"Extracted cognitive memories: {len(memories)}")
                
            # Process preference extraction
            preferences = self.process_preference_extraction_batch(current_batch)
            self.logger.debug(f"Extracted preferences: {preferences}")
            
            # Only summarize if we have more remaining messages than max_messages setting
            if len(remaining_messages) >= self.settings.llm_config["chatbot_agent"]["max_messages"]:
                # Summarize the batch and replace messages
                summary, deleted_messages = self.summarize_and_replace_batch(current_batch)
                state.messages.extend(deleted_messages)
                all_summaries.append(summary)
                self.logger.debug(f"Batch summarized and replaced")
            else:
                self.logger.debug(f"Skipping summarization since only {len(remaining_messages)} messages remain")

        
        state.summaries.extend(all_summaries)
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

    def process_cognitive_memory_batch(self, messages: List[BaseMessage]) -> List[CognitiveMemory]:
        """
        Process a batch of messages to extract cognitive memories.
        
        Args:
            messages (List[BaseMessage]): Batch of messages to process
            
        Returns:
            List[CognitiveMemory]: List of extracted cognitive memories
            
        Logs:
            DEBUG: Processing details and extracted memories
            INFO: Batch processing status
            ERROR: Any extraction or storage failures
        """
        max_retries = 3
        retry_count = 0
        examples = get_cognitive_memory_examples()
        # Get existing memory types for context
        with driver.session() as session:
            memory_index = session.execute_read(self.get_memory_index)
        
        # Convert messages to conversation text
        conversation_text = prepare_conversation_data(messages)
        
        while retry_count < max_retries:
            try:
                self.logger.debug(f"Processing cognitive memory batch with {len(messages)} messages")
                
                # Create prompt variables
                prompt_variables = {
                    "conversation_text": conversation_text,
                    "existing_types": json.dumps(memory_index.model_dump()),
                    "current_time": datetime.utcnow().isoformat(),
                    "examples": examples
                }
                
                # Extract memories
                llm_output = self.cognitive_memory_extractor_chain.invoke(prompt_variables)
                
                # Convert LLM output to list if it's not already
                memory_items = [llm_output] if not isinstance(llm_output, list) else llm_output
                
                # Convert items to CognitiveMemory objects if needed
                memories = []
                for item in memory_items:
                    try:
                        if isinstance(item, CognitiveMemory):
                            memories.append(item)
                        elif isinstance(item, dict):
                            # Convert nested dictionaries to their respective models
                            if 'semantic' in item and isinstance(item['semantic'], dict):
                                item['semantic'] = SemanticAttributes(**item['semantic'])
                            if 'temporal' in item and isinstance(item['temporal'], dict):
                                item['temporal'] = TemporalContext(**item['temporal'])
                            if 'validation' in item and isinstance(item['validation'], dict):
                                item['validation'] = ValidationStatus(**item['validation'])
                            if 'relations' in item and isinstance(item['relations'], dict):
                                item['relations'] = MemoryRelations(**item['relations'])
                            
                            # Create CognitiveMemory object
                            memory = CognitiveMemory(**item)
                            memories.append(memory)
                        else:
                            self.logger.warning(f"Unexpected memory type: {type(item)}")
                            continue
                    except Exception as e:
                        self.logger.error(f"Error converting memory to model: {str(e)}")
                        continue
                
                # Process and store each memory
                stored_memories = []
                with driver.session() as session:
                    for memory in memories:
                        try:
                            # Generate embedding if not present
                            if memory.embedding is None:
                                embedding_text = memory.to_embedding_text()
                                memory.embedding = self.embeddings_service.generate_embedding(embedding_text)
                            
                            # Store the memory
                            db_id = session.execute_write(create_cognitive_memory_node, memory)
                            
                            # Update the memory index
                            session.execute_write(
                                self.update_memory_index,
                                memory.type,
                                memory.content
                            )
                            
                            stored_memories.append(memory)
                            self.logger.debug(f"Stored cognitive memory: {memory.type} - {memory.id}")
                            
                        except Exception as e:
                            self.logger.error(f"Error storing cognitive memory: {str(e)}")
                            continue
                
                return stored_memories
                
            except ValidationError as ve:
                retry_count += 1
                self.logger.error(f"ValidationError on attempt {retry_count}: {ve}")
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached. Unable to process memory batch.")
                    raise
                continue
                
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred: {e}")
                raise
                
    def get_memory_index(self,tx) -> MemoryIndex:
        """
        Retrieve the current memory type index from Neo4j.
        
        Args:
            tx: Neo4j transaction object
            
        Returns:
            MemoryIndex: Current index of memory types
        """
        query = """
        MATCH (m:CognitiveMemory)
        RETURN m.type as type, m.content as content
        """
        
        result = tx.run(query)
        
        # Build the index
        index = MemoryIndex()
        for record in result:
            mem_type = record["type"]
            content = record["content"]
            
            # Update counts
            index.type_counts[mem_type] = index.type_counts.get(mem_type, 0) + 1
            
            # Update examples
            if mem_type not in index.type_examples:
                index.type_examples[mem_type] = []
            if len(index.type_examples[mem_type]) < 3:  # Keep up to 3 examples
                index.type_examples[mem_type].append(content)
                
        return index

    def update_memory_index(self,tx, memory_type: str, content: str):
        """
        Update the memory type index with a new memory.
        
        Args:
            tx: Neo4j transaction object
            memory_type: Type of the memory
            content: Content of the memory
        """
        query = """
        MERGE (i:MemoryIndex {type: $type})
        ON CREATE SET 
            i.count = 1,
            i.examples = [$content]
        ON MATCH SET
            i.count = i.count + 1,
            i.examples = CASE 
                WHEN size(i.examples) < 3 THEN i.examples + [$content]
                ELSE i.examples
            END
        """
        
        tx.run(query, type=memory_type, content=content)

    def _create_relationship(self, tx: ManagedTransaction, relationship: Relationship) -> None:
        """
        Create a relationship between two entities in Neo4j.

        Args:
            tx (ManagedTransaction): The Neo4j transaction
            relationship (Relationship): The relationship object to be created

        Raises:
            Exception: If there is an error creating the relationship

        Logs:
            DEBUG: Relationship creation details
            ERROR: Any errors during creation
        """
        try:
            self.logger.debug(f"Creating relationship of type {relationship.type} between entities {relationship.from_entity_id} and {relationship.to_entity_id}")
            create_entity_relationship(tx, relationship)
        except Exception as e:
            self.logger.error(f"Error creating relationship: {str(e)}")
            raise

if __name__ == "__main__":
    
    def main():
        """
        Main async function to initialize and run the memory agent.
        
        Logging:
            - Info for start/completion
            - Error for any exceptions
        """
        logger.info("Initializing memory agent")
        try:
            # Initialize the memory agent
            memory_agent = SmileMemory()

            # Create a test AgentState with some messages
            state = {"messages": []}
            
            config = {
                "configurable": {
                    "thread_id": settings.app_config["langchain_config"]["thread_id"], 
                    "checkpoint_id": settings.app_config["langchain_config"]["checkpoint_id"]
                }
            }
            
            # Await the async execute_graph call
            response = memory_agent.execute_graph(state=state, config=config)
            logger.info("Memory agent execution completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise

    main()


