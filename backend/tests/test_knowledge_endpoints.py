"""
Tests for the knowledge graph endpoints from 'backend/app/api/routers.py'.
This module tests the graph-related endpoints including node fetching, relationships, search, and exploration.

Detailed logging is used to trace behaviors and potential issues.
"""

import pytest
import logging
from fastapi import HTTPException
from typing import Dict, List, Any, Optional
from app.services.neo4j.driver import driver
from app.services.embeddings import EmbeddingsService

# Import router endpoint functions
from app.api.routers import (
    get_nodes,
    fetch_relationships,
    semantic_search,
    get_graph_data,
    explore_graph
)

from app.utils.logger import logger

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Known node IDs and properties for testing
TEST_USER_ID = "4:ab101d43-a5c0-462d-8a09-ca41bd1666e1:9"  # Eric Lampron
TEST_PERSON_ID = "4:ab101d43-a5c0-462d-8a09-ca41bd1666e1:3"  # Marie
TEST_ORG_ID = "4:ab101d43-a5c0-462d-8a09-ca41bd1666e1:20"  # Thinkmax
TEST_MEMORY_ID = "25dc3e5f-497d-4138-b1ac-378fb9d8e94b"

class TestKnowledgeNodes:
    """Tests for node-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_user_node(self):
        """Test fetching User node."""
        try:
            response = await get_nodes(node_type="Person")  # Eric is stored as a Person
            assert isinstance(response, list)
            assert len(response) > 0
            
            # Find the specific user
            user = next((node for node in response if node["id"] == TEST_USER_ID), None)
            assert user is not None
            assert user["name"] == "Eric Lampron"
            
            logger.info("[TEST] Successfully tested get_nodes for User type")
        except Exception as e:
            logger.error(f"Error testing get_nodes for User: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_get_person_node(self):
        """Test fetching Person node."""
        try:
            response = await get_nodes(node_type="Person")
            assert isinstance(response, list)
            assert len(response) > 0
            
            # Find the specific person
            person = next((node for node in response if node["id"] == TEST_PERSON_ID), None)
            assert person is not None
            assert person["name"] == "Marie"
            assert person["type"] == "Person"
            
            logger.info("[TEST] Successfully tested get_nodes for Person type")
        except Exception as e:
            logger.error(f"Error testing get_nodes for Person: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_get_organization_node(self):
        """Test fetching Organization node."""
        try:
            response = await get_nodes(node_type="Organization")
            assert isinstance(response, list)
            assert len(response) > 0
            
            # Find the specific organization
            org = next((node for node in response if node["id"] == TEST_ORG_ID), None)
            assert org is not None
            assert org["name"] == "Thinkmax"
            assert org["industry"] == "IT Services"
            assert "Workplace of Eric Lampron" in org["notes"]
            
            logger.info("[TEST] Successfully tested get_nodes for Organization type")
        except Exception as e:
            logger.error(f"Error testing get_nodes for Organization: {str(e)}")
            raise

class TestKnowledgeRelationships:
    """Tests for relationship-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_fetch_relationships(self):
        """Test fetching relationships for a specific entity."""
        try:
            # Use Eric Lampron's node ID since we know he has relationships
            node_id = TEST_USER_ID
            response = await fetch_relationships(node_id=node_id)
            
            assert isinstance(response, list)
            assert len(response) > 0
            
            # Verify relationship structure
            for rel in response:
                assert "id" in rel
                assert "type" in rel
                assert "source" in rel
                assert "target" in rel
                assert "properties" in rel
                assert "source_node" in rel
                assert "target_node" in rel
            
            # Verify we can find relationships involving Eric Lampron
            assert any(
                "Eric Lampron" in str(rel["source_node"].get("name", "")) or 
                "Eric Lampron" in str(rel["target_node"].get("name", ""))
                for rel in response
            )
            
            logger.info("Successfully tested fetch_relationships endpoint")
        except Exception as e:
            logger.error(f"Error testing fetch_relationships: {str(e)}")
            raise

class TestKnowledgeSearch:
    """Tests for search-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_person(self):
        """Test semantic search for person-related information."""
        try:
            response = await semantic_search(
                query="Eric Lampron",
                node_type="Person",
                limit=5,
                min_score=0.5
            )
            
            assert isinstance(response, list)
            assert len(response) > 0
            # We expect to find nodes related to Eric Lampron
            assert any("Eric" in str(node.get("name", "")) for node in response)
            
            logger.info("[TEST] Successfully tested semantic_search for person")
        except Exception as e:
            logger.error(f"Error testing semantic_search for person: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_semantic_search_organization(self):
        """Test semantic search for organization-related information."""
        try:
            response = await semantic_search(
                query="Thinkmax IT Services",
                node_type="Organization",
                limit=5,
                min_score=0.5
            )
            
            assert isinstance(response, list)
            assert len(response) > 0
            # We expect to find the Thinkmax organization
            assert any("Thinkmax" in str(node.get("name", "")) for node in response)
            
            logger.info("[TEST] Successfully tested semantic_search for organization")
        except Exception as e:
            logger.error(f"Error testing semantic_search for organization: {str(e)}")
            raise

class TestKnowledgeExploration:
    """Tests for graph exploration endpoints."""
    
    @pytest.mark.asyncio
    async def test_explore_graph_person(self):
        """Test graph exploration with person-related search."""
        try:
            response = await explore_graph(
                search="Eric Lampron",
                node_type="Person",
                limit=100
            )

            assert isinstance(response, dict)
            assert "nodes" in response
            assert "relationships" in response
            assert len(response["nodes"]) > 0
            
            # Verify we can find Eric Lampron in the results
            assert any("Eric Lampron" in str(node["properties"].get("name", "")) 
                      for node in response["nodes"])
            
            logger.info("[TEST] Successfully tested explore_graph for person")
        except Exception as e:
            logger.error(f"Error testing explore_graph for person: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_explore_graph_organization(self):
        """Test graph exploration with organization-related search."""
        try:
            response = await explore_graph(
                search="Thinkmax",
                node_type="Organization",
                limit=100
            )

            assert isinstance(response, dict)
            assert "nodes" in response
            assert "relationships" in response
            assert len(response["nodes"]) > 0
            
            # Verify we can find Thinkmax in the results
            assert any("Thinkmax" in str(node["properties"].get("name", "")) 
                      for node in response["nodes"])
            
            logger.info("[TEST] Successfully tested explore_graph for organization")
        except Exception as e:
            logger.error(f"Error testing explore_graph for organization: {str(e)}")
            raise

    @pytest.mark.asyncio
    async def test_get_graph_data_all_types(self):
        """Test fetching graph data for all node types."""
        try:
            response = await get_graph_data(
                node_types=["User", "Person", "Organization", "CognitiveMemory"],
                depth=2,
                limit=100
            )

            assert isinstance(response, dict)
            assert "nodes" in response
            assert "relationships" in response
            assert len(response["nodes"]) > 0
            
            # Verify we can find known nodes
            node_names = [str(node["properties"].get("name", "")) for node in response["nodes"]]
            assert any("Eric Lampron" in name for name in node_names)
            assert any("Thinkmax" in name for name in node_names)
            
            logger.info("[TEST] Successfully tested get_graph_data for all types")
        except Exception as e:
            logger.error(f"Error testing get_graph_data for all types: {str(e)}")
            raise 
