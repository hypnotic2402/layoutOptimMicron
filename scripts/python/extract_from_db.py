from neo4j import GraphDatabase

def extract():
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "12345678")

    mcs = []

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()

        macros,summ , keys = driver.execute_query("Match (n) Return n")
        e,s,k = driver.execute_query("Match ()-[r]->() Return r")
        if len(macros) > 0:
            print(type(macros[0]))
        else:
            print("DB Empty")
        i = 0
        for macro in macros:
            mcs.append(macro)
            print("Macro " + str(i))
            print(macro)
            i+=1

    
    nMacros = len(mcs)
    nodes = [m[0] for m in mcs]
    macIds = [m[0].element_id for m in mcs]
    wh = []
    for m in mcs:
        wh.append(m[0].get('w'))
        wh.append(m[0].get('h'))

    # print(e[0][0].nodes)
    edges = [ed[0].nodes for ed in e]
    nl = []

    for ed in edges:
        nl.append([macIds.index(ed[0].element_id) , macIds.index(ed[1].element_id)])
        # print(ed[1])


    # print(nl)

    return nMacros , wh , nl