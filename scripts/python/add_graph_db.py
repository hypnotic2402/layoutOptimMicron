from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

spi_file_path = '/home/iiitb_micron/micron/github/layoutOptimMicron/examples/netlists/four_inv.spi'

class bbox:

    def __init__(self , name):
        self.name = name
        self.typeName = ""
        self.ports = []
        self.x = 0
        self.y = 0
        self.connectedTo = []
        self.idx = 0


def parse_spi_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    subcircuits = set()
    connections = []
    topCircuit = ""
    for line in lines:
        if line.startswith('* Cell Name  :'):
            topCircuit = line.split()[4]
            print("tc = " + topCircuit)
        if line.startswith('.subckt'):
            subcircuit_name = line.split()[1]
            subcircuits.add(subcircuit_name)
        
    tcr = False
    lc = 0
    tcLines = []
    for line in lines:

        if line.startswith('.subckt'):
            subcircuit_name = line.split()[1]
            if subcircuit_name == topCircuit:
                tcr = True
        
        if line.startswith('.ends'):
            subcircuit_name = line.split()[1]
            if subcircuit_name == topCircuit:
                tcr = False

        if tcr:
            lc+=1
            tcLines.append(line)

    tc_bboxes = []

    for line in tcLines:
        if line.startswith('X'):
            print(line)
            lx =  line.split()
            bbName = lx[0]
            tc_bboxes.append(bbox(bbName))
            typeIdx = 0
            for i in range(len(lx)):
                if lx[i][0] == '$':
                    typeIdx = i-1
                    break
            tc_bboxes[-1].typeName = lx[typeIdx]
            for i in range(1,typeIdx):
                tc_bboxes[-1].ports.append(lx[i])
            for i in range(len(lx)):
                if lx[i][0:2] == '$X':
                    tc_bboxes[-1].x = lx[i][3:]
                if lx[i][0:2] == '$Y':
                    tc_bboxes[-1].y = lx[i][3:]


    for i in range(len(tc_bboxes)):
        tc_bboxes[i].idx = i     
    
    for bb1 in tc_bboxes:
        for bb2 in tc_bboxes:
            if (bb2.name not in bb1.connectedTo) and (bb1.name != bb2.name):
                
                for i in bb1.ports:
                    if ((i != 'vdd') and (i != 'vss')):
                        print(i)
                        if i in bb2.ports:
                            bb1.connectedTo.append(bb2.idx)
                            bb2.connectedTo.append(bb1.idx)
                            break
    

    adjMatrix = np.zeros([len(tc_bboxes) , len(tc_bboxes)])
    
    for i in range(len(tc_bboxes)):
        for j in tc_bboxes[i].connectedTo:
            adjMatrix[i,j] = 1

    print(adjMatrix)
    return subcircuits, connections , adjMatrix , tc_bboxes


def draw_circuit_graph(subcircuits, connections):
    G = nx.Graph()
    G.add_nodes_from(subcircuits)
    G.add_edges_from(connections)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_weight='bold', font_size=10, edge_color='gray')
    plt.show()

def draw_netlist(tc_bboxes , adj):
    G = nx.Graph()
    for bb in tc_bboxes:
        G.add_node(bb.name)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if (adj[i,j] == 1):
                G.add_edge(tc_bboxes[i].name , tc_bboxes[j].name)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_weight='bold', font_size=10, edge_color='gray')
    plt.show()


subcircuits, connections , adjMatrix , tc_bboxes = parse_spi_file(spi_file_path)
draw_netlist(tc_bboxes , adjMatrix)


with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

    # Converting to cypher instructions
    mCount = 0
    for bb in tc_bboxes:
        instr = "CREATE (macro" + str(bb.idx) + ": Macro {name:'" + str(bb.name) + "' , x: " + str(bb.x) + " , y: " + str(bb.y) + "})\n"
        ret = driver.execute_query(instr)
        mCount += 1
    wCount = 0
    edgesAdded = []
    for i in range(adjMatrix.shape[0]):
        for j in range(adjMatrix.shape[1]):
            if (adjMatrix[i, j] == 1):
                if ([tc_bboxes[i].idx , tc_bboxes[j].idx] not in edgesAdded):
                    edgesAdded.append([tc_bboxes[i].idx , tc_bboxes[j].idx]  )
                    edgesAdded.append([tc_bboxes[j].idx, tc_bboxes[i].idx])
                    instr = "CREATE (macro" + str(tc_bboxes[i].idx) + ")-[w" + str(wCount) + ":IS_CONNECTED]->(macro"+str(tc_bboxes[j].idx)+")\n"
                    ret = driver.execute_query(instr)
                    wCount+=1
