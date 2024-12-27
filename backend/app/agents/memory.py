import asyncio
from datetime import datetime
import json
import logging
import os
from uuid import UUID, uuid4
from typing import List, Dict, Optional, Tuple
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
    MemoryRelations,
    CognitiveAspect,
    MemorySource
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

logger = logging.getLogger(__name__)

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
        self.embeddings_service = EmbeddingsService(driver)
        
        # Initialize schemas
        with driver.session() as session:
            try:
                initialize_schema_with_session(session)
                logger.info("Successfully initialized cognitive memory schema")
            except Exception as e:
                logger.error(f"Failed to initialize schema: {str(e)}")
                raise
      

    def initialize(self):
        """Initialize PostgreSQL checkpointer and set up tables."""
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

    def save_memory(self, memory: CognitiveMemory) -> CognitiveMemory:
        """
        Save a single cognitive memory directly to the knowledge graph.
        This method bypasses the langraph workflow and allows direct memory creation.
        
        Args:
            memory (CognitiveMemory): The memory to save
            
        Returns:
            CognitiveMemory: The saved memory with updated database information
            
        Raises:
            ValueError: If memory validation fails
            Exception: If there's an error storing the memory
            
        Logs:
            DEBUG: Memory storage details
            ERROR: Any storage failures
        """
        try:
            self.logger.debug(f"Saving cognitive memory: {memory.type} - {memory.id}")
            
            # Generate embedding if not present
            if memory.embedding is None:
                embedding_text = memory.to_embedding_text()
                memory.embedding = self.embeddings_service.generate_embedding(embedding_text)
            
            # Store the memory
            with driver.session() as session:
                # Store the memory and get the Neo4j node ID
                neo4j_node_id = session.execute_write(create_cognitive_memory_node, memory)
                
                # Update the memory index
                session.execute_write(
                    self.update_memory_index,
                    memory.type,
                    memory.content
                )
                
                self.logger.debug(f"Successfully stored cognitive memory: {memory.type} - {memory.id} (Neo4j ID: {neo4j_node_id})")
                return memory
                
        except ValidationError as ve:
            self.logger.error(f"ValidationError while saving memory: {ve}")
            raise ValueError(f"Memory validation failed: {ve}")
            
        except Exception as e:
            self.logger.error(f"Error saving cognitive memory: {str(e)}")
            raise

    def get_context(self, query: str, chat_id: str = None) -> str:
        """
        Get contextual information based on query using a unified knowledge retrieval approach.
        
        Args:
            query (str): The input query to find relevant context
            chat_id (str, optional): Chat session identifier
            
        Returns:
            str: Formatted context string
            
        Example response format:
            Contextual information for your understanding:
            
            PINNED KNOWLEDGE:
            - Preference (Score: 0.98)
              content: User prefers dark mode for all applications
              importance: 0.9
              last_updated: 2024-03-15
            
            RELEVANT KNOWLEDGE:
            - CognitiveMemory (Score: 0.95)
              content: Meeting with John about project timeline
              type: FACTUAL
              importance: 0.8
              creation_date: 2024-03-14
            - ConversationSummary (Score: 0.92)
              content: Discussion about family history
              topics: ["family", "history"]
              participants: ["user", "assistant"]
        """
        logger.debug(f"Getting context for query: {query}")
        
        # Define vector indices and their corresponding labels
        knowledge_sources = [
            {"label": "CognitiveMemory", "index_name": "memory_embeddings"},
            {"label": "Preference", "index_name": "preference_vector"},
            {"label": "ConversationSummary", "index_name": "summary_vector"},
            {"label": "Person", "index_name": "person_vector"},
            {"label": "Organization", "index_name": "org_vector"},
            {"label": "Document", "index_name": "document_vector"}
        ]
        
        # Collect relevant nodes from all sources
        relevant_nodes = []
        for source in knowledge_sources:
            try:
                nodes = self._search_vector_store(
                    query=query,
                    index_name=source["index_name"],
                    label=source["label"]
                )
                relevant_nodes.extend(nodes)
                logger.debug(f"Found {len(nodes)} nodes from {source['label']}")
            except Exception as e:
                logger.error(f"Error searching {source['label']}: {str(e)}")
                continue
            
        # Rerank all nodes together based on relevance
        reranked_nodes = self._rerank_nodes(query, relevant_nodes)
        
        # Separate into regular and pinned knowledge
        pinned_knowledge = [n for n in reranked_nodes if n.get("node", {}).get("is_pinned", False)]
        regular_knowledge = [n for n in reranked_nodes if not n.get("node", {}).get("is_pinned", False)]
        
        # Format context
        context = "Contextual information for your understanding:\n"
        
        if pinned_knowledge:
            context += "\nPINNED KNOWLEDGE:\n"
            context += self._format_knowledge_section(pinned_knowledge)
            
        if regular_knowledge:
            context += "\nRELEVANT KNOWLEDGE:\n"
            context += self._format_knowledge_section(regular_knowledge)
            
        logger.debug(f"Generated context with {len(pinned_knowledge)} pinned and {len(regular_knowledge)} relevant items")
        return context

    def _search_vector_store(self, query: str, index_name: str, label: str) -> List[Dict]:
        """
        Search a specific vector store for relevant nodes.
        
        Args:
            query (str): Search query
            index_name (str): Name of the vector index
            label (str): Neo4j label to search
            
        Returns:
            List[Dict]: List of relevant nodes with their properties and scores
        """
        logger.debug(f"Searching vector store: {index_name} for label: {label}")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_service.generate_embedding(query)
            
            cypher = """
            CALL db.index.vector.queryNodes($index_name, 10, $embedding)
            YIELD node, score
            WHERE $label IN labels(node)
            WITH node, score, labels(node) as labels
            OPTIONAL MATCH (node)-[:BELONGS_TO]->(u:Person)
            WITH node, score, labels, u,
                 CASE 
                    WHEN node.importance >= 0.8 THEN true
                    WHEN node.is_pinned = true THEN true
                    ELSE false
                 END as is_pinned
            RETURN node {
                .*, 
                is_pinned: is_pinned,
                labels: labels,
                user: CASE WHEN u IS NOT NULL THEN u.name ELSE null END
            } as node, 
            score, 
            labels
            LIMIT 5
            """
            
            with driver.session() as session:
                result = session.run(
                    cypher,
                    index_name=index_name,
                    embedding=query_embedding,
                    label=label
                )
                
                nodes = []
                for record in result:
                    node_data = dict(record["node"])
                    nodes.append({
                        "node": node_data,
                        "score": record["score"],
                        "source_index": index_name
                    })
                
                logger.debug(f"Found {len(nodes)} nodes in {index_name}")
                return nodes
                
        except Exception as e:
            logger.error(f"Error searching vector store {index_name}: {str(e)}")
            raise

    def _rerank_nodes(self, query: str, nodes: List[Dict]) -> List[Dict]:
        """
        Rerank nodes based on semantic similarity and other factors.
        
        Args:
            query (str): Original search query
            nodes (List[Dict]): List of nodes with their scores
            
        Returns:
            List[Dict]: Reranked list of nodes
        """
        try:
            # Sort by score and apply additional ranking factors
            reranked = sorted(nodes, key=lambda x: (
                x.get("score", 0),  # Primary sort by vector similarity
                x.get("node", {}).get("importance", 0),  # Secondary sort by importance
                x.get("node", {}).get("access_count", 0)  # Tertiary sort by access count
            ), reverse=True)
            
            logger.debug(f"Reranked {len(nodes)} nodes")
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking nodes: {str(e)}")
            return nodes  # Return original order if reranking fails

    def _format_knowledge_section(self, nodes: List[Dict]) -> str:
        """
        Format knowledge nodes into readable sections.
        
        Args:
            nodes (List[Dict]): List of knowledge nodes to format
            
        Returns:
            str: Formatted string representation
        """
        formatted = ""
        for node in nodes:
            node_data = node["node"]
            
            # Get the primary label (excluding base labels like 'Node')
            node_type = next((label for label in node_data.get("labels", []) 
                            if label not in ["Node"]), "Unknown")
            
            # Format properties (excluding technical fields)
            excluded_props = {
                "embedding", "vector", "id", "labels", "isPinned",
                "created_at", "updated_at", "access_count"
            }
            
            properties = {k: v for k, v in node_data.items() 
                        if k not in excluded_props and not k.startswith("_")}
            
            # Start with type and score
            formatted += f"- {node_type} (Score: {node['score']:.2f})\n"
            
            # Add properties in a readable format
            for prop, value in properties.items():
                # Format property name to be more readable
                prop_name = prop.replace("_", " ").title()
                
                # Format value based on type
                if isinstance(value, (list, set)):
                    formatted += f"  {prop_name}: {', '.join(map(str, value))}\n"
                elif isinstance(value, dict):
                    formatted += f"  {prop_name}:\n"
                    for k, v in value.items():
                        formatted += f"    {k}: {v}\n"
                else:
                    formatted += f"  {prop_name}: {value}\n"
            
            formatted += "\n"
            
        return formatted

if __name__ == "__main__":
    
    def main():
        """
        Main function to test memory agent functionality.
        
        Logging:
            - Info for start/completion
            - Error for any exceptions
        """
        logger.info("Initializing memory agent")
        try:
            # Initialize the memory agent
            memory_agent = SmileMemory()

            # Test direct memory creation
            test_memory = CognitiveMemory(
                type="FACTUAL",
                content="This is a test memory",
                semantic=SemanticAttributes(
                    cognitive_aspects=[CognitiveAspect.FACTUAL, CognitiveAspect.BEHAVIORAL]
                ),
                temporal=TemporalContext(
                    observed_at=datetime.utcnow(),
                    valid_from=datetime.utcnow()
                ),
                validation=ValidationStatus(
                    is_valid=True,
                    validation_source=MemorySource.DIRECT_OBSERVATION,
                    validation_timestamp=datetime.utcnow()
                ),
                relations=MemoryRelations(
                    related_entities=[],
                    related_summaries=[],
                    related_preferences=[],
                    associations=[]
                ),
                access_count=0,
                version=1
            )

            # Save the test memory
            saved_memory = memory_agent.save_memory(test_memory)
            logger.info(f"Test memory saved successfully: {saved_memory.id}")

            # Test the langraph workflow as well
            state = {"messages": []}
            config = {
                "configurable": {
                    "thread_id": settings.app_config["langchain_config"]["thread_id"], 
                    "checkpoint_id": settings.app_config["langchain_config"]["checkpoint_id"]
                }
            }
            
            # Run the graph workflow
            response = memory_agent.execute_graph(state=state, config=config)
            logger.info("Memory agent execution completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise

    main()


