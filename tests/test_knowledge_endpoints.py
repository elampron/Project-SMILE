import pytest
from neo4j import GraphDatabase

# Assuming driver is a global instance of GraphDatabase.driver

@pytest.fixture
async def test_relationship(test_node):
    """Create a test relationship for testing."""
    source_id = await test_node
    with driver.session() as session:
        # Create target node
        target_result = session.execute_write(
            lambda tx: tx.run(
                "CREATE (n:TestNode {name: 'Related Node'}) RETURN elementId(n) as node_id"
            ).single()
        )
        target_id = target_result["node_id"]
        
        # Create relationship
        rel_result = session.execute_write(
            lambda tx: tx.run(
                "MATCH (s), (t) WHERE elementId(s) = $source_id AND elementId(t) = $target_id "
                "CREATE (s)-[r:TEST_REL]->(t) RETURN elementId(r) as rel_id",
                source_id=source_id,
                target_id=target_id
            ).single()
        )
        rel_id = rel_result["rel_id"]
        
        data = {"source_id": source_id, "target_id": target_id, "rel_id": rel_id}
        
        # Cleanup
        try:
            yield data
        finally:
            session.execute_write(
                lambda tx: tx.run(
                    "MATCH (s)-[r]->(t) WHERE elementId(r) = $rel_id DELETE r",
                    rel_id=rel_id
                )
            )
            session.execute_write(
                lambda tx: tx.run(
                    "MATCH (n) WHERE elementId(n) = $node_id DELETE n",
                    node_id=target_id
                )
            )