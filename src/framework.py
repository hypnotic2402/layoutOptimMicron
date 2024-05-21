import classes as cls
import placement
import routing
import placement2

class Framework:
    def __init__(self , macros , nets , floor):
        self.macros = macros
        self.nets = nets
        self.floor = floor
        # self.PS = placement.PlacementSolver(self.macros, self.nets, self.floor , verb=False)
        self.PS = placement2.PlacementSolver(self.macros, self.nets , self.floor , 1000 , margin=60)
        self.RS = routing.RoutingSolver(self.macros, self.nets , self.floor)

    def place(self, iter , genVid=0 , filename=None , verbose=False):
        # self.PS = placement.PlacementSolver(self.macros, self.nets, self.floor ,verb=True , pop=1000 )
        self.PS = placement2.PlacementSolver(self.macros, self.nets , self.floor , 1000, margin=60)
        self.PS.place(iter)
        if genVid == 1:
            self.PS.genVid(filename)

    def importPlacement(self, xy): #xy = [x1,y1,x2,y2,...]
        for i in range(int(len(xy)/2)):
            self.macros[i].x = xy[2*i]
            self.macros[i].y = xy[(2*i)+1]

    def route(self , disp=False):
        self.RS = routing.RoutingSolver(self.macros, self.nets , self.floor)
        nts = self.RS.route(self.RS.Nets)
        for i in range(len(self.nets)):
            self.nets[i].routed_cells = nts[i].routed

        if disp:
            self.RS.display_curr_matr(self.RS.Nets , 2)


    
        




if __name__ == '__main__':
    M1 = cls.Macro("m1" , 0 , 100 , 50 , [])
    P1M1 = cls.Pin(0,0)
    M1.pins.append(P1M1)
    M2 = cls.Macro("m2" , 1 ,110 , 40, [])
    P2M1 = cls.Pin(0,0)
    M2.pins.append(P2M1)
    M3 = cls.Macro("m3" , 2 ,60 , 60, [])
    P3M1 = cls.Pin(0,0)
    M3.pins.append(P3M1)
    N1 = cls.Net('n1' , [M1 , M2])
    N2 = cls.Net('n2' , [M2 , M3])

    macros = [M1 , M2 , M3]
    nets = [N1 , N2]
    
    FL = cls.Floor(800,800,1)
    FW = Framework(macros , nets , FL)

    print("___________Placement__________")
    FW.place(1000 , genVid=1 , filename="testVid5.avi")
    

    # FW.importPlacement([0,0,500,500,0,400])
    # for macro in macros :
    #     print(macro.x)
    #     print(macro.y)

    print("_________Routing_________")
    FW.route(disp=False)
    

    

    


