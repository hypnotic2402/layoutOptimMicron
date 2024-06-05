import classes as cls
import numpy as np
from pymoo.core.problem import Problem
# from pymoo.algorithms.soo.nonconvex.ga import GA
from scripts.python.ga2 import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import math
from operator import add
from pymoo.core.callback import Callback
import numpy
import cv2
import time

k1 = 1
k2 = 1

def hpwl(XX, wh , nets,macros):
    s = 0
    res = []
    for X in XX:
        for net in nets:
            xmin = math.inf
            xmax = -math.inf
            ymin = math.inf
            ymax = -math.inf
            macros_net = net.macros
            for macro in macros_net:
                index = next((i for i, item in enumerate(macros) if item.id == macro.id), -1)
                # print(len(X))
                x = X[2*index]
                y = X[2*index+1]
                if (x > xmax):
                    xmax = x
                if (x < xmin):
                    xmin = x
                if (y > ymax):
                    ymax = y
                if (y < ymin):
                    ymin = y

            s += xmax - xmin + ymax - ymin
        res.append(s)

    return res

def overlapAr(X , wh, citr,itr):
    for i in range(int(len(X) / 2)):
        xi_min = X[2*i]
        yi_min = X[2*i + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        for j in range(int(len(X) /2)):
            if (i != j):
                xj_min = X[2*j]
                yj_min = X[2*j + 1]
                xj_max = xj_min + wh[2*j]
                yj_max = yj_min + wh[2*j + 1]
            
                dx = min(xi_max , xj_max) - max(xi_min , xj_min)
                dy = min(yi_max , yj_max) - max(yi_min , yj_min)

                if (dx >= 0) and (dy >= 0):
                    return dx*dy

    return 0

def isOverlapping(XX , wh):
    
    res = []
    for X in XX:
        res.append(overlapAr(X , wh))   
    return res

def objF(X , nets , wh , macros):
    f = list(map(add , [k1*x for x in hpwl(X, wh , nets , macros)] , [k2*x for x in isOverlapping(X , wh)]))
    f = [ -x for x in f]
    return f



class MyProblem(Problem):
    def __init__(self  , n_var , xl , xu , nets, wh , macros):
        super().__init__(n_var=n_var , n_obj=1 , xl=xl , xu=xu)
        self.nets = nets
        self.wh = wh
        self.macros = macros

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = objF(x , self.nets , self.wh , self.macros)


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("X"))

class PlacementSolver:
    def __init__(self , macros , nets , floor ,verb , pop=100):
        self.macros = macros
        self.nets = nets
        self.floor = floor
        self.wh = []
        for macro in macros:
            self.wh.append(macro.w)
            self.wh.append(macro.h)
        self.prob = MyProblem((2*len(self.macros)) , 0 , self.floor.h - max(self.wh) , self.nets , self.wh , self.macros)

        self.algo = GA(pop_size=pop , eliminate_duplicates=True)
        self.pRes = []
        self.verb = verb

    def run(self , iter):
        termination = get_termination("n_gen", iter)
        res = minimize(self.prob, self.algo,
               seed=1,
               verbose=self.verb,termination=termination,callback=MyCallback())
        val = res.algorithm.callback.data["best"]
        idxMin = [numpy.argmin(objF(v , self.nets , self.wh , self.macros)) for v in val]
        ret = []
        for (v,idx) in zip(val,idxMin):
            ret.append(v[idx])
        self.pRes = ret


    def genVid(self , filename):
        out = cv2.VideoWriter(filename,0,40, (self.floor.h,self.floor.h))
        for frameX in self.pRes:
            img1 = np.zeros((800, 800, 3), np.uint8)
            X = frameX
            for i in range(int(len(X) / 2)):
                xi_min = X[2*i]
                yi_min = X[2*i + 1]
                xi_max = xi_min + self.wh[2*i]
                yi_max = yi_min + self.wh[2*i + 1]
                cv2.rectangle(img1, (int(xi_min) , int(yi_min)) , (int(xi_max) , int(yi_max)) , (0,0,255) , 3)

            out.write(img1)

        cv2.destroyAllWindows()  
        out.release()

    def place(self, iter):
        self.run(iter)
        fin = self.pRes[-1]
        for i in range(int(len(fin)/2)):
            self.macros[i].x = fin[2*i]
            self.macros[i].y = fin[(2*i) + 1]


if __name__=="__main__":

    FL = cls.Floor(800,800,1)
    M1 = cls.Macro("m1" , 0 , 100 , 50 , [2,3])
    M2 = cls.Macro("m2" , 1 ,110 , 40, [2,3])
    M3 = cls.Macro("m3" , 2 ,60 , 60, [2,3])
    N1 = cls.Net('n1' , [M1 , M2])
    N2 = cls.Net('n2' , [M2 , M3])

    macros = [M1 , M2 , M3]
    nets = [N1 , N2]

    PS = PlacementSolver(macros , nets , FL, 1)
    PS.run(20)
    print(PS.pRes[-1])
    PS.genVid('placement3.avi')


