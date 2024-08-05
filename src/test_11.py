import random
import classes as cls
import routing
import time
import math
from framework import Framework
import cProfile

def test_all_macros_connected(macros, nets):
    for macro in macros:
        connected = any(macro in net.macros for net in nets)
        assert connected, f"Macro {macro.name} is not connected to any net"

def round_up_to_next_hundred(n):
    return math.ceil(n / 100.0) * 100

if __name__ == '__main__':

    MACRO_CNT = random.randint(5, 75)
    WIDTH_MAX = 150
    WIDTH_MIN = 50
    HEIGHT_MAX = 150
    HEIGHT_MIN = 50
    PIN_PER_MACRO=1
    # NET_CNT = 20
    RAND_NET_CNT = 1 # Probability of a macro being connected to a net

    random.seed(42)  # For reproducibility of random numbers

    macros = []
    tot_area=0
    for i in range(MACRO_CNT):
        width = random.randint(WIDTH_MIN, WIDTH_MAX)
        height = random.randint(HEIGHT_MIN, HEIGHT_MAX)
        tot_area+=width*height
        macro_id = i
        macro_name = f"m{macro_id+1}"
        macro = cls.Macro(macro_name, macro_id, width, height, [])
        # for _ in range(PIN_PER_MACRO):  # Assuming we want to generate 50 random pins on edges
        #     if random.choice([True, False]):
        #         # Pin on vertical edges (x = 0 or x = width)
        #         x = random.choice([0, width])
        #         y = random.randint(0, height)
        #         # y=random.choice([0,height])
        #     else:
        #         # Pin on horizontal edges (y = 0 or y = height)
        #         y = random.choice([0, height])
        #         x = random.randint(0, width)
        #         # x=random.choice([0,width])
        #     if cls.Pin(x,y) not in macro.pins:
        #         macro.pins.append(cls.Pin(x, y))
        
        macros.append(macro)
    avg_area=(tot_area)/MACRO_CNT
    print("Area Occupied by Macros is:",tot_area)
    gamma=MACRO_CNT/8
    square_floor_plan=round_up_to_next_hundred(math.sqrt(avg_area*(gamma*MACRO_CNT)))
    print("For sqaure shaped floor plan:",square_floor_plan)
    inp=int(input("For custom floor plan press 1, for square shaped floor plan press 2:"))
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
        if (len(nets) == 100): break
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
        net = cls.Net(net_name, [macro1, macro2])
        nets.append(net)

    # Ensure all macros are connected to at least one net
    # test_all_macros_connected(macros, nets)
    
    print("Number of macros: ", len(macros))
    print("Number of nets: ", len(nets))

    FL = cls.Floor(math.ceil(w), math.ceil(h), 30)
    FW = Framework(macros, nets, FL)
    start_time=time.time()
    print("___________Placement__________")
    FW.place(100, genVid=0, filename=f"images/10MACROS_{len(macros)}_{len(nets)}_NSGA2_{FW.floor.w}*{FW.floor.h}.png")
    end_time=time.time()
    print("Placement Time: ", end_time-start_time)
    start_time = time.time()
    macros = FW.macros
    for macro in macros:
        print(f"{macro.name} : ({macro.x}, {macro.y})")  
    
    for net in nets:
        print(f"{net.name} : {[(macro.name, macro.x, macro.y) for macro in net.macros]}")


    print("_________Routing_________")
    FW.route(disp=True)
    end_time = time.time()
    print("Routing Time: ", end_time - start_time)

