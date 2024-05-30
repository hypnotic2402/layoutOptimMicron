import classes as cls
import framework as fw
from neo4j import GraphDatabase
from extract import extract as extr, create_nodes
import time
import Scripts 

if __name__ == '__main__':

    # created nodes from the cypher output file from the parser

    Scripts.main(folder_path="../examples/netlists_text_files", num_designs=6)

    create_nodes(
        cypher_file_path="output.cypher",
    )

    # Extract from DB and required files


    nMacros , macros , nets = extr()
    for net in nets:
        print(net.macros[0].w, net.macros[1].h)
    # macDict = {}
    for macro in macros:
        macro.pins.append(cls.Pin(0,0))
        # print("Macro: ", macro.id, " Pins: ", macro.pins, " Width: ", macro.w, " Height: ", macro.h, " X: ", macro.x, " Y: ", macro.y)
    # for net in nets:
    #     if net.macros[0].id not in macDict:
    #         macDict[net.macros[0].id] = 1
    #     else: macDict[net.macros[0].id] += 1
    #     if net.macros[1].id not in macDict:
    #         macDict[net.macros[1].id] = 1
    #     else: macDict[net.macros[1].id] += 1

    # for key, vals in macDict.items():
    #     for i in range(vals):
    #         macros[key].pins.append(cls.Pin(i, 0))


    
    FL = cls.Floor(800,800,20)


    # Push to framework : macros, nets, floor

    FW = fw.Framework(macros , nets, FL)

    # Place

    start_time=time.time()
    print("___________Placement__________")
    FW.place(10, genVid=0, filename="images/10MACROS.png")
    end_time=time.time()
    print("Placement Time: ", end_time-start_time)
    # for macro in macros:
    #     macro.pins.append(cls.Pin(0,0))
    #     print("Macro: ", macro.id, " Pins: ", macro.pins, " Width: ", macro.w, " Height: ", macro.h, " X: ", macro.x, " Y: ", macro.y)
    

    # Route

    start_time = time.time()
    print("_________Routing_________")
    FW.route(disp=True)
    end_time = time.time()
    
    print("Routing Time: ", end_time - start_time)

