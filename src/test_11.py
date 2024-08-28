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

def generate_pin(macro):
    while True:
        if random.choice([True, False]):
            # Pin on vertical edges (x = 0 or x = width)
            x = random.choice([0, macro.w])
            y = random.randint(0, macro.h)
            # y=random.choice([0,height])
        else:
            # Pin on horizontal edges (y = 0 or y = height)
            y = random.choice([0, macro.h])
            x = random.randint(0, macro.w)
            # x=random.choice([0,width])
        if cls.Pin(x,y) not in macro.pins:
            break
    return cls.Pin(x, y)



if __name__ == '__main__':

    MACRO_CNT_MIN=10
    MACRO_CNT_MAX=100
    # MACRO_CNT = random.randint(MACRO_CNT_MIN, MACRO_CNT_MAX)
    MACRO_CNT=50
    WIDTH_MAX = 150
    WIDTH_MIN = 50
    HEIGHT_MAX = 150
    HEIGHT_MIN = 50
    PIN_MAX=20
    PIN_MIN=10
    PIN_PER_MACRO=random.randint(PIN_MIN, PIN_MAX)
    random.seed(42)  

    macros = []

    tot_area=0 #total area occupied by all macros

    for i in range(MACRO_CNT):
        width = random.randint(WIDTH_MIN, WIDTH_MAX)
        height = random.randint(HEIGHT_MIN, HEIGHT_MAX)

        tot_area+=width*height

        macro_id = i
        macro_name = f"m{macro_id+1}"
        macro = cls.Macro(macro_name, macro_id, width, height, [])
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

    nets = []
    LIMIT_NET_CNT = 2*MACRO_CNT
    # Generate nets
    for i in range(LIMIT_NET_CNT):
        if len(nets) == LIMIT_NET_CNT:
            break
        num_macros_in_net = random.randint(2, min(MACRO_CNT,15))  # Randomly choose number of macros in the net
        selected_macros = random.sample(macros, num_macros_in_net)
        
        net_name = f"n{len(nets) + 1}"
        pins_and_macro=[]
        for macro in selected_macros:
            pin=generate_pin(macro)
            macro.pins.append(pin)
            pins_and_macro.append((pin, macro))

        net=cls.Net(net_name, list(selected_macros),pins_and_macro)
        # print("Net pins: ", net.pins)
        nets.append(net)

    # Ensure all macros are connected to at least one net
    # test_all_macros_connected(macros, nets)
    for macro in macros:
        if len(macro.pins)==0:
            #Generate a pin for the macro
            pin=generate_pin(macro)
            macro.pins.append(pin)
            #select any random net from existing nets and add the macro to it
            net = random.choice(nets)
            net.macros.append(macros[i])
            net.pins.append((macros[i].pins[0],macros[i]))

    
    print("Number of macros: ", len(macros))
    print("Number of nets: ", len(nets))
    for macro in macros:
        print(f"{macro.name} : ({macro.x}, {macro.y}) and width:{macro.w} and height:{macro.h}")      
        for pin in macro.pins:
            print(f"{macro.name} : ({pin.x}, {pin.y})")
    
    for net in nets:
        print(f"{net.name} : {[(macro.name, pin.x, pin.y) for pin, macro in net.pins]}")

    FL = cls.Floor(math.ceil(w), math.ceil(h), 1)
    FW = Framework(macros, nets, FL)
    start_time=time.time()
    print("___________Placement__________")
    FW.place(100, genVid=0, filename=f"images/10MACROS_{len(macros)}_{len(nets)}_NSGA2_{FW.floor.w}*{FW.floor.h}.png")
    end_time=time.time()
    print("Placement Time: ", end_time-start_time)
    start_time = time.time()
    macros = FW.macros


    print("_________Routing_________")
    FW.route(disp=True)
    end_time = time.time()
    print("Routing Time: ", end_time - start_time)

