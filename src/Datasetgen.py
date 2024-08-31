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
    


def genData(iterations):
    def round_up_to_next_hundred(n):
        return math.ceil(n / 100.0) * 100

    if sys.argv[0] == "--file":
        pass
    else:
        MACRO_CNT = 2*random.randint(10, 50)
        WIDTH_MAX = 150
        WIDTH_MIN = 50
        HEIGHT_MAX = 150
        HEIGHT_MIN = 50
        NUM_NETS = MACRO_CNT*5
        LAMBDA = 1
        PIN_PER_MACRO = 1

        MAX_NET_CNT_PER_MACRO = 10
  
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
    gamma=MACRO_CNT/8
    square_floor_plan=round_up_to_next_hundred(math.sqrt(avg_area*(gamma*MACRO_CNT)))
    w=square_floor_plan
    h=square_floor_plan
    
    net_set = set()
    net_avail = set(macros)
    net_cnt = {}
    nets = []
    edge_list = []
    net_cnt_list_per_macro = []
    mean = random.randint(1, 5)
    std = random.randint(1, max(1, mean // 2))
    for i in range(len(macros)):
        # fix the number of nets using a gaussian distribution
        NUM_NETS = min(max(1, int(np.random.normal(mean, std))), MAX_NET_CNT_PER_MACRO)
        net_cnt_list_per_macro.append(NUM_NETS)
        
        for _ in range(int(NUM_NETS)):
            macro1 = macros[i]
            macro2 = random.choice(list(net_avail))
            while macro2 == macro1:
                macro2 = random.choice(list(net_avail))

            net_set.add(macro1)
            net_set.add(macro2)
            edge_list.append((macro1, macro2))
            net_name = f"n{len(nets)+1}"
            if macro1.id not in net_cnt.keys(): net_cnt[macro1.id] = 1 
            else: net_cnt[macro1.id] += 1
            if macro2.id not in net_cnt.keys(): net_cnt[macro2.id] = 1 
            else: net_cnt[macro2.id] += 1
            if net_cnt[macro1.id] == MAX_NET_CNT_PER_MACRO: net_avail.remove(macro1)
            if net_cnt[macro2.id] == MAX_NET_CNT_PER_MACRO: net_avail.remove(macro2)
            net = cls.Net(net_name, [macro1, macro2],[(macro1.pins[0],macro1),(macro2.pins[0],macro2)])
            nets.append(net)
    
    print(f"Number of Macros: {len(macros)}")
    print(f"Number of Nets: {len(nets)}")
    print("Starting placement...")

    FL = cls.Floor(math.ceil(w), math.ceil(h), LAMBDA)
    FW = Framework(macros, nets, FL)
    start_time=time.time()
    total_loss, hpwl_sm, overlap = FW.place(iterations, genVid=0, gen_image=0)
    end_time = time.time()
    hpwl = []
    for edge in edge_list:
        macro1 = FW.macros[edge[0].id]
        macro2 = FW.macros[edge[1].id]
        hpwl.append(abs(macro1.x - macro2.x) + abs(macro1.y - macro2.y))
    
    assert len(hpwl) == len(edge_list)
    return macros, edge_list, hpwl, FL

def run_test_n_times(n = 1, iterations=200):
    FILENAME = 'dataset4.csv'
    start_csv(file_path=FILENAME)
    for i in range(n):
        macros, edges, hpwl, FL = genData(iterations)
        write_to_csv(macros, edges, hpwl, FL, file_path=FILENAME)
        
    


if __name__ == "__main__":
    datapoints = 1000
    iterations = 200
    run_test_n_times(datapoints, iterations)