from neo4j import GraphDatabase
import networkx as nx
from karateclub import Node2Vec


def get_embedding():

    URI = "neo4j://localhost"
    AUTH = ("neo4j", "12345678")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    query = """
    MATCH (n)-[r]->(c) RETURN *
    """

    results = driver.session().run(query)

    G = nx.MultiDiGraph()

    nodes = list(results.graph()._nodes.values())
    for node in nodes:
        G.add_node(node.id, labels=node._labels, properties=node._properties)

    rels = list(results.graph()._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

    model = Node2Vec(epochs=1000)
    model.fit(G)

    x = model.get_embedding()
    print(x)

    return x

if __name__=="__main__":
    get_embedding()