import numpy as np
import random
import math
import classes as cls
import cv2
import time
from logger import Logger

class Individual:
    def __init__(self , xu , xl , yu , yl):
        self.x = random.uniform(xl, xu)
        self.y = random.uniform(yl, yu)
        

class DE: 
    def __init__(self , pop_size , mutation_factor, crossover_prob , objF , x_len , xl, xu , nets , wh ,margin=0):
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.pop = []
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.nets = nets
        self.wh = [x+margin for x in wh]
        self.ranked_pop = []
        self.best_pop = []
        self.xHis = []
        self.logger = Logger.getInstance()

        for p in range(self.pop_size):
            px = [random.uniform(self.xl, self.xu) for _ in range(self.x_len)]
            self.pop.append(px)

        for p in self.pop:
            loss = objF(p, self.wh, self.nets)
            self.ranked_pop.append((loss[0], p, loss[1], loss[2]))
        self.ranked_pop.sort(reverse=True)
        self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_prob))]

    def mutate(self, target_idx):
        candidates = list(range(self.pop_size))
        candidates.remove(target_idx)
        a, b, c = random.sample(candidates, 3)
        mutant = [self.pop[a][i] + self.mutation_factor * (self.pop[b][i] - self.pop[c][i]) for i in range(self.x_len)]
        return [self.bound(mutant[i]) for i in range(self.x_len)]

    def bound(self, value):
        return min(max(value, self.xl), self.xu)

    def crossover(self, target, mutant):
        crossovered = [mutant[i] if random.random() < self.crossover_prob else target[i] for i in range(self.x_len)]
        return crossovered

    def opt(self, n_iter):
        for iter in range(n_iter):
            if iter % 10 == 0: 
                self.logger.log(f"Iteration {iter} ----> l:{self.ranked_pop[0][0]} h:{self.ranked_pop[0][2]}  o:{self.ranked_pop[0][3]}")
                print(f"Iteration {iter} ----> l:{self.ranked_pop[0][0]} h:{self.ranked_pop[0][2]}  o:{self.ranked_pop[0][3]}")
            self.xHis.append(self.ranked_pop[0])

            new_pop = []
            for i in range(self.pop_size):
                target = self.pop[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                new_pop.append(trial if self.objF(trial, self.wh, self.nets)[0] > self.objF(target, self.wh, self.nets)[0] else target)

            self.pop = new_pop
            self.ranked_pop = []

            for p in self.pop:
                loss = self.objF(p , self.wh , self.nets)
                self.ranked_pop.append((loss[0] , p, loss[1], loss[2]))
            self.ranked_pop.sort(reverse=True)

            self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_prob))]
        
        isOverlapping(self.xHis[-1][1], self.wh)
        x_min, y_min, x_max, y_max, tot_area = getBoundingBox(self.xHis[-1][1], self.wh)

class PlacementSolver:
    floor = None
    def __init__(self, macros , nets , floor , pop_size , margin=0):
        self.macros = macros
        self.Nets = nets
        self.floor = floor
        PlacementSolver.floor = floor
        self.nets = []
        self.margin = margin
        for net in self.Nets:
            self.nets.append([x.id for x in net.macros])
        self.wh = []
        for macro in macros:
            self.wh.append(macro.w)
            self.wh.append(macro.h)
        self.de = DE(pop_size , 0.5 , 0.7 , objF , 2*len(self.macros) , 0 ,self.floor.h - max(self.wh) , self.nets , self.wh,1)

    def place(self , iter):
        self.de.opt(iter)
        X = self.de.xHis[-1][1]
        for i in range(len(self.macros)):
            self.macros[i].x = min(X[(2*i)], self.floor.w - self.wh[2*i])
            self.macros[i].y = min(X[(2*i) + 1], self.floor.h - self.wh[2*i + 1])

    def genVid(self ,path, full_video=False):
        if not full_video: 
            img1 = np.zeros((self.floor.h,self.floor.w, 3), np.uint8)
            X = self.de.xHis[-1][1]
            for i in range(int(len(X) / 2)):
                xi_min = self.macros[i].x
                yi_min = self.macros[i].y
                xi_max = xi_min + self.wh[2*i]
                yi_max = yi_min + self.wh[2*i + 1]
                cv2.rectangle(img1, (int(xi_min) , int(yi_min)) , (int(xi_max) , int(yi_max)) , (0,0,255) , 3)
                cv2.putText(img1, f"{self.macros[i].name}", (int(xi_min), int(yi_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imwrite(path, img1)
        else:
            out = cv2.VideoWriter(path,0,40, (self.floor.w,self.floor.h))
            for frameX in self.de.xHis:
                img1 = np.zeros((self.floor.w,self.floor.h, 3), np.uint8)
                X = frameX[1]
                for i in range(int(len(X) / 2)):
                    xi_min = X[2*i]
                    yi_min = X[2*i + 1]
                    xi_max = xi_min + self.wh[2*i]
                    yi_max = yi_min + self.wh[2*i + 1]
                    cv2.rectangle(img1, (int(xi_min) , int(yi_min)) , (int(xi_max) , int(yi_max)) , (0,0,255) , 3)
                out.write(img1)
            cv2.destroyAllWindows()  
            out.release()

def hpwl(X , wh , nets):
    s = 0
    for net in nets:
        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf
        for macroIdx in net:
            x = X[2*macroIdx]
            y = X[2*macroIdx+1]
            if (x > xmax):
                xmax = x
            if (x < xmin):
                xmin = x
            if (y > ymax):
                ymax = y
            if (y < ymin):
                ymin = y
        s += xmax - xmin + ymax - ymin
    return s

def hpwlFaster(X, wh, nets):
    X = np.array(X).reshape(-1, 2)
    s = 0
    for net in nets:
        net_coords = X[net]
        xmin, ymin = net_coords.min(axis=0)
        xmax, ymax = net_coords.max(axis=0)
        s += xmax - xmin + ymax - ymin
    return s

def getBoundingBox(X, wh):
    X = np.array(X).reshape(-1, 2)
    wh = np.array(wh).reshape(-1, 2)
    x_min, y_min = X[:, 0].min(), X[:, 1].min()
    x_max, y_max = (X + wh)[:, 0].max(), (X + wh)[:, 1].max()
    tot_area = (wh[:, 0] * wh[:, 1]).sum()
    return x_min, y_min, x_max, y_max, tot_area

def isOverlapping(X , wh):
    total_overlapp = 0
    for i in range(int(len(X) / 2)):
        xi_min = X[2*i]
        yi_min = X[2*i + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        for j in range(i,int(len(X) /2)):
            if (i != j):
                xj_min = X[2*j]
                yj_min = X[2*j + 1]
                xj_max = xj_min + wh[2*j]
                yj_max = yj_min + wh[2*j + 1]
            
                dx = min(xi_max , xj_max) - max(xi_min , xj_min)
                dy = min(yi_max , yj_max) - max(yi_min , yj_min)
                if (dx>=0) and (dy>=0):
                    total_overlapp += dx*dy
    return total_overlapp

def objF(X , wh , nets):
    overlapp=isOverlapping(X , wh)
    hpwl_val = hpwlFaster(X , wh , nets)
    # print("Overlapping: ", overlapp, " HPWL: ", hpwl_val)
    # return (- 1/len(X) * hpwl_val - 10000*overlapp*len(X), hpwl_val, overlapp)
    
    X_np = np.array(X).reshape(-1, 2)
    wh_np = np.array(wh).reshape(-1, 2)
    # give function to calculate the eucledian sum of distances of all macros from the center of the floor plan
    # x_sm = np.sum(np.abs(X_np[:, 0] - (wh[0] / 2)))
    # y_sm = np.sum(np.abs(X_np[:, 1] - (wh[1] / 2)))

    W, H = PlacementSolver.floor.w, PlacementSolver.floor.h

    x_center, y_center = wh_np[:,0] / 2, wh_np[:,1] / 2
    y_top = X_np[:, 1] + y_center
    y_bottom = W - (X_np[:, 1] - y_center) 
    x_left = X_np[:, 0] + x_center
    x_right = H - (X_np[:, 0] - x_center)
    y_top, y_bottom, x_left, x_right = y_top**2, y_bottom**2, x_left**2, x_right**2

    xy_tot = np.sum(np.minimum(np.minimum(np.minimum(x_left, x_right), y_top), y_bottom))
    # y_sm = np.sum(np.minimum(y_top, y_bottom))

    return (- 1 * hpwl_val - (len(X) ** 2) * overlapp - (len(X)) * (xy_tot), hpwl_val, overlapp)

