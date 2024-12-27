from app.services.neo4j import driver

def main():
    """Check Neo4j indices."""
    with driver.session() as session:
        result = session.run('SHOW INDEXES')
        for record in result:
            print(record)

if __name__ == "__main__":
    main() 