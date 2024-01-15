from neo4j import GraphDatabase

URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

def clear_db(driver):
    driver.execute_query("match (a) -[r] -> () delete a, r")
    driver.execute_query("match (a) delete a")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    clear_db(driver)
