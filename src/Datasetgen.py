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
import csv


def start_csv(file_path = 'dataset.csv'):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Macros', 'Netlist', 'HPWL'])

def write_to_csv(macros, edges, hpwl, FL, file_path='dataset.csv'):
    macro_list = [(mac.w, mac.h, (mac.x / FL.w) * 100, (mac.y / FL.h) * 100) for mac in macros]
    edge_list = [(mac1.id, mac2.id) for mac1, mac2 in edges]

    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([macro_list, edge_list, hpwl])
    


def genData():
    def round_up_to_next_hundred(n):
        return math.ceil(n / 100.0) * 100

    if sys.argv[0] == "--file":
        pass
    else:
        MACRO_CNT = 2*random.randint(10, 35)
        WIDTH_MAX = 150
        WIDTH_MIN = 50
        HEIGHT_MAX = 150
        HEIGHT_MIN = 50
        NUM_NETS = MACRO_CNT/2
        LAMBDA = 1
        PIN_PER_MACRO = 1

        # NET_CNT = 20
        RAND_NET_CNT = 1 # Probability of a macro being connected to a net

        # random.seed(42)  # For reproducibility of random numbers

        macros = []
        tot_area=0
        for i in range(MACRO_CNT):
            width = random.randint(WIDTH_MIN, WIDTH_MAX)
            height = random.randint(HEIGHT_MIN, HEIGHT_MAX)
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
    edge_list = []
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
        edge_list.append((macro1, macro2))
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
    FW = Framework(macros, nets, FL)
    start_time=time.time()
    total_loss, hpwl_sm, overlap = FW.place(200, genVid=0, filename=f"images/10MACROS_{len(macros)}_{len(nets)}_testhighOL_{FW.floor.w}*{FW.floor.h}.png")
    end_time = time.time()
    hpwl = []
    for edge in edge_list:
        # print(edge[0].id, edge[1].id)
        macro1 = FW.macros[edge[0].id]
        macro2 = FW.macros[edge[1].id]
        hpwl.append(abs(macro1.x - macro2.x) + abs(macro1.y - macro2.y))
    
    assert len(hpwl) == len(edge_list)

    print("Time: ", end_time-start_time)

    return macros, edge_list, hpwl, FL

def run_test_n_times(n = 1):
    FILENAME = 'dataset1.csv'
    start_csv(file_path=FILENAME)
    for i in range(n):
        macros, edges, hpwl, FL = genData()
        write_to_csv(macros, edges, hpwl, FL, file_path=FILENAME)
        
    


if __name__ == "__main__":
    n = 500
    run_test_n_times(n)