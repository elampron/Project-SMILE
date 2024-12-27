"""
Knowledge search service for advanced semantic search across all node types in Neo4j.

This module provides advanced knowledge search capabilities including:
1. Query augmentation using LLM
2. Multi-query strategy execution
3. Result reranking
4. Unified search across all node types
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import openai
from datetime import datetime
from neo4j import GraphDatabase, Session
from app.services.embeddings import EmbeddingsService
from app.services.neo4j import vector_similarity_search, get_related_entities
from app.configs.settings import Settings

# Initialize logger
logger = logging.getLogger(__name__)

class KnowledgeSearchService:
    """
    Service for advanced semantic search across all node types in Neo4j.
    Provides query augmentation, multi-query strategy, and result reranking.
    """
    
    def __init__(self, driver: GraphDatabase.driver, embeddings_service: Optional[EmbeddingsService] = None):
        """
        Initialize the KnowledgeSearchService.
        
        Args:
            driver: Neo4j driver instance
            embeddings_service: Optional EmbeddingsService instance. If not provided, a new one will be created.
        """
        self.driver = driver
        self.embeddings_service = embeddings_service or EmbeddingsService()
        self.logger = logging.getLogger(__name__)
        
        # Get LLM config
        settings = Settings()
        llm_config = settings.llm_config.get("chatbot_agent", {})
        self.model = llm_config.get("params", {}).get("model", "gpt-4")
        
        # Define node types and their corresponding vector indexes
        self.node_types = {
            "Preference": "preference_vector",
            "Summary": "summary_vector", 
            "Person": "person_vector",
            "Organization": "org_vector",
            "Document": "document_vector",
            "CognitiveMemory": "memory_embeddings"
        }
    
    def search_knowledge(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.7,
        conversation_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using semantic search.
        
        Args:
            query: Natural language query
            filters: Optional filters to apply (e.g. node types, date ranges)
            limit: Maximum number of results per type
            min_score: Minimum similarity score threshold
            conversation_context: Optional conversation context to help with query augmentation
            
        Returns:
            List of relevant nodes with their properties and scores
        """
        # Generate embedding for query
        query_vector = self.embeddings_service.generate_embedding(query)
        
        results = []
        
        # Search each node type
        with self.driver.session() as session:
            for node_type, index_name in self.node_types.items():
                try:
                    # Apply any type-specific filters
                    additional_filters = None
                    if filters and "node_types" in filters:
                        if node_type not in filters["node_types"]:
                            continue
                    
                    type_results = vector_similarity_search(
                        session=session,
                        index_name=index_name,
                        query_vector=query_vector,
                        k=limit,
                        min_score=min_score,
                        additional_filters=additional_filters
                    )
                    
                    # Add node type to results
                    for result in type_results:
                        results.append({
                            "node": result["node"],
                            "score": result["score"],
                            "node_label": node_type
                        })
                        
                except Exception as e:
                    logger.warning(f"Error searching node type {node_type}: {e}")
                    continue
                    
        # Sort all results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top results up to limit
        return results[:limit]
    
    def _augment_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple search queries based on the original query.
        
        Args:
            query: Original natural language query
            conversation_context: Optional context from recent conversation
            
        Returns:
            List[str]: List of augmented queries
        """
        # For now, just return the original query
        # In the future, we can use LLM to generate variations
        return [query]

if __name__ == "__main__":
    """
    Test script for the KnowledgeSearchService
    """
    import logging
    from neo4j import GraphDatabase
    from app.configs.settings import Settings
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load settings
    settings = Settings()
    
    # Initialize Neo4j driver
    driver = GraphDatabase.driver(
        settings.app_config["neo4j_config"]["uri"],
        auth=(
            settings.app_config["neo4j_config"]["username"],
            settings.app_config["neo4j_config"]["password"]
        )
    )
    
    try:
        # Initialize services
        knowledge_service = KnowledgeSearchService(driver)
        
        # Create test data
        with driver.session() as session:
            # Create test preference node
            session.run("""
                CREATE (n:Preference)
                SET n = $properties
                """,
                properties={
                    "id": "test-pref-1",
                    "preference_type": "work_schedule",
                    "details": "Preferred hours: 9-5 EST",
                    "importance": 4,
                    "created_at": "2024-01-01",
                    "is_test": True,
                    "embedding": knowledge_service.embeddings_service.generate_embedding(str({
                        "preference_type": "work_schedule",
                        "details": "Preferred hours: 9-5 EST",
                        "importance": 4
                    }))
                }
            )
            
            # Create test memory node
            session.run("""
                CREATE (n:CognitiveMemory)
                SET n = $properties
                """,
                properties={
                    "id": "test-memory-1",
                    "content": "Client expressed concerns about project deadlines",
                    "importance": 0.8,
                    "created_at": "2024-01-01",
                    "is_test": True,
                    "embedding": knowledge_service.embeddings_service.generate_embedding(
                        "Client expressed concerns about project deadlines"
                    )
                }
            )
            
            logger.info("Created test data in Neo4j")
            
            # Test queries
            test_queries = [
                {
                    "query": "What are my work schedule preferences?",
                    "expected_types": ["Preference"]
                },
                {
                    "query": "Any concerns about project deadlines?",
                    "expected_types": ["CognitiveMemory"]
                }
            ]
            
            # Run test queries
            for test in test_queries:
                logger.info("\n" + "=" * 50)
                logger.info(f"Testing query: {test['query']}")
                logger.info(f"Expected node types: {test['expected_types']}")
                
                try:
                    results = knowledge_service.search_knowledge(
                        query=test["query"],
                        filters={"node_types": test["expected_types"]},
                        limit=5
                    )
                    
                    if results:
                        logger.info(f"\nFound {len(results)} results:")
                        for result in results:
                            logger.info(f"\nType: {result['node_label']}")
                            logger.info(f"Score: {result['score']}")
                            logger.info(f"Content: {result['node']}")
                    else:
                        logger.info("No results found")
                        
                except Exception as e:
                    logger.error(f"Error processing query '{test['query']}': {str(e)}")
            
            # Clean up test data
            session.run("MATCH (n) WHERE n.is_test = true DETACH DELETE n")
            logger.info("\nCleaned up test data")
            
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")
    finally:
        driver.close()
        logger.info("Tests completed") 