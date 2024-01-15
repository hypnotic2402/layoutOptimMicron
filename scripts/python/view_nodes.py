from neo4j import GraphDatabase

URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

    macros,summ , keys = driver.execute_query("Match ((n)-[r]->(m)) Return n,r,m")
    if len(macros) > 0:
        print(type(macros[0]))
    else:
        print("DB Empty")
    i = 0
    for macro in macros:
        print("Macro " + str(i))
        print(macro)
        i+=1
