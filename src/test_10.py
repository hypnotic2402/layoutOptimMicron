import random
import classes as cls
import routing
import time
import math
from framework import Framework
import cProfile
from logger import Logger
from os import sys
import numpy as np
import csvwriter

def test_all_macros_connected(macros, nets):
    for macro in macros:
        connected = any(macro in net.macros for net in nets)
        assert connected, f"Macro {macro.name} is not connected to any net"

def round_up_to_next_hundred(n):
    return math.ceil(n / 100.0) * 100


def main(): 

    "Args: iter-> for Datasetgen.py, shows which iteration is running"


    if sys.argv[0] == "--file":
        pass
    else:
        logger=Logger.getInstance("prelog1.txt")
        # MACRO_CNT = 2*random.randint(10, 35)
        MACRO_CNT=100
        WIDTH_MAX = 150
        WIDTH_MIN = 50
        HEIGHT_MAX = 150
        HEIGHT_MIN = 50
        NUM_NETS = MACRO_CNT+10
        LAMBDA = 1
        PIN_PER_MACRO = 1

        # NET_CNT = 20
        RAND_NET_CNT = 1 # Probability of a macro being connected to a net

        random.seed(42)  # For reproducibility of random numbers

        macros = []
        tot_area=0
        for i in range(MACRO_CNT):
            width = random.randint(WIDTH_MIN, WIDTH_MAX)
            height = random.randint(HEIGHT_MIN, HEIGHT_MAX)
            logger.log(f"Macro Number:{i+1}, width:{width},height:{height}")
            tot_area+=width*height
            macro_id = i
            macro_name = f"m{macro_id+1}"
            macro = cls.Macro(macro_name, macro_id, width, height, [])
            pin = cls.Pin(0, 0)
            macro.pins.append(pin)
            macros.append(macro)
    avg_area=(tot_area)/MACRO_CNT
    print("Area Occupied by Macros is:",tot_area)
    gamma=MACRO_CNT/8
    square_floor_plan=round_up_to_next_hundred(math.sqrt(avg_area*(gamma*MACRO_CNT)))
    print("For sqaure shaped floor plan:",square_floor_plan)
    # inp=int(input("For custom floor plan press 1, for square shaped floor plan press 2:"))
    inp=2
    if inp==1:
        w=int(input("Enter width of floor plan:"))
        h=int(input("Enter height of floor plan:"))
    else:  
        w=square_floor_plan
        h=square_floor_plan
    net_set = set()
    net_avail = set(macros)
    net_cnt = {}
    nets = []
    LIMIT_NET_CNT = 1
    for i in range(len(macros)):
        if (random.random() < RAND_NET_CNT and macros[i] in net_set and macros[i] not in net_avail): continue
        if (len(nets) == NUM_NETS): break
        macro1 = macros[i]
        while True:
            macro2 = random.choice(list(net_avail))
            if macro2 != macro1:
                break
        net_set.add(macro1)
        net_set.add(macro2)
        net_name = f"n{len(nets)+1}"
        if macro1.id not in net_cnt.keys(): net_cnt[macro1.id] = 1 
        else: net_cnt[macro1.id] += 1
        if macro2.id not in net_cnt.keys(): net_cnt[macro2.id] = 1 
        else: net_cnt[macro2.id] += 1
        if net_cnt[macro1.id] == LIMIT_NET_CNT: net_avail.remove(macro1)
        if net_cnt[macro2.id] == LIMIT_NET_CNT: net_avail.remove(macro2)
        net = cls.Net(net_name, [macro1, macro2],[(macro1.pins[0],macro1),(macro2.pins[0],macro2)])
        nets.append(net)

    

    
    print("Number of macros: ", len(macros))
    print("Number of nets: ", len(nets))
    FL = cls.Floor(math.ceil(w), math.ceil(h), LAMBDA)

    MODULE_NAME = "PlacementNSGA2"
    FW = Framework(macros, nets, FL,population=3000)
    start_time=time.time()
    print("___________Placement__________")
    FW.place(
        iter=600, 
        genVid=0, 
        gen_image=1,
        filename=f"images/10MACROS_{len(macros)}_{len(nets)}_{MODULE_NAME}_{FW.floor.w}*{FW.floor.h}.png",
        )
    
    end_time=time.time()
    print("Placement Time: ", end_time-start_time)
    for net in nets:
        logger.log(f"Net {net.name} connected between {net.macros[0].name} and {net.macros[1].name}")
        for pin, macro in net.pins:
            print(f"Pin {pin.x},{pin.y} of {macro.name}")
            logger.log(f"Pin {pin.x},{pin.y} of {macro.name}")
    start_time = time.time()
    macros = FW.macros
    for macro in macros:
        logger = Logger.getInstance()
        logger.log(f"{macro.name} : ({macro.x}, {macro.y})")
        print(f"{macro.name} : ({macro.x}, {macro.y})") 

    print("_________Routing_________")
    FW.route(disp=True)
    end_time = time.time()
    print("Routing Time: ", end_time - start_time)

    csvwriter.csvwriter(iter)


if __name__ == '__main__':
    main()
