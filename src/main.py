import classes as cls
import framework as fw
from neo4j import GraphDatabase
from extract import extract as extr

if __name__ == '__main__':

    # Extract from DB and required files
    
    P3M1 = cls.Pin(0,0)
    P4M1 = cls.Pin(0,0)
    P1M1 = cls.Pin(0,0)
    P2M1 = cls.Pin(0,0)

    # M1 = cls.Macro("m1" , 0 , 100 , 50 , [])
    
    # M1.pins.append(P1M1)
    # M2 = cls.Macro("m2" , 1 ,110 , 40, [])
    
    # M2.pins.append(P2M1)
    # M3 = cls.Macro("m3" , 2 ,60 , 60, [])
    
    # M3.pins.append(P3M1)
    # N1 = cls.Net('n1' , [M1 , M2])
    # N2 = cls.Net('n2' , [M2 , M3])

    # macros = [M1 , M2 , M3]
    # nets = [N1 , N2]

    nMacros , macros , nets = extr()
    macros[0].pins.append(P1M1)
    macros[1].pins.append(P2M1)
    macros[2].pins.append(P3M1)
    macros[3].pins.append(P4M1)
    
    FL = cls.Floor(800,800,10)

    print(nets[0].macros[0].pins)

    # Push to framework : macros, nets, floor


    FW = fw.Framework(macros , nets, FL)

    # Place

    print("Placing Macros")
    FW.place(iter=1000,genVid=0 , filename="testVid4.avi")
    print("Placement Done")

    # Route

    print("Routing Nets")
    FW.route(disp=True)
    print("Routing Done")

    # Update to DB



