import classes as cls
from neo4j import GraphDatabase

def extract(uri = "neo4j://localhost" , auth = ("neo4j", "12345678")):
    # URI = "neo4j://localhost"
    # AUTH = ("neo4j", "12345678")

    mcs = []

    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()

        macros,summ , keys = driver.execute_query("Match (n) Return n")
        e,s,k = driver.execute_query("Match ()-[r]->() Return r")
        if len(macros) > 0:
            print("Extracting from DB")
        else:
            print("DB Empty")
        i = 0
        for macro in macros:
            mcs.append(macro)
            # print("Macro " + str(i))
            # print(macro)
            i+=1

    nMacros = len(mcs)
    nodes = [m[0] for m in mcs]
    macIds = [m[0].element_id for m in mcs]
    wh = []
    for m in mcs:
        wh.append(m[0].get('w'))
        wh.append(m[0].get('h'))
    mcrs = []
    i = 0
    for m in mcs:
        # print("HI")
        # print(m[0].get('w'))
        mcrs.append(cls.Macro("M" + str(macIds[i]) , i , int(m[0].get('w')) , int(m[0].get('h')) , []))
        # print(mcrs[-1].w)
        i+=1
    edges = [ed[0].nodes for ed in e]
    nl = []

    for ed in edges:
        nl.append([macIds.index(ed[0].element_id) , macIds.index(ed[1].element_id)])

    nets = []
    i =0
    for i in range(len(nl)):
        nets.append(cls.Net("N"+str(i) , [mcrs[id] for id in nl[i]]))
        i+=1

    return nMacros , mcrs , nets

if __name__=="__main__":
    print(extract())