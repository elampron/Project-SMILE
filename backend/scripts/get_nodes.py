"""Script to query node IDs from the database."""

from app.services.neo4j.driver import driver

def main():
    with driver.session() as session:
        result = session.run('''
            MATCH (n)
            WHERE n:User OR n:Person OR n:Organization
            RETURN labels(n) as label, elementId(n) as id, n.name as name, n.type as type
            LIMIT 10
        ''')
        for record in result:
            print(f"{record['label']}: {record['id']} - {record['name']} ({record.get('type', '')})")

if __name__ == "__main__":
    main() 