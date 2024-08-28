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
        

class GA: 
    def __init__(self , pop_size , mutation_factor, crossover_factor , objF , x_len , xl, xu , nets , wh ,margin=0):
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_factor = crossover_factor
        self.pop = []
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.ranked_pop = []
        self.best_pop = []
        self.wh = wh
        self.nets = nets
        self.xHis = []
        self.logger = Logger.getInstance()
        self.wh = [x+margin for x in wh]

        

        for p in range(self.pop_size):
            px = []
            for x in range(self.x_len):
                px.append(random.uniform(self.xl,self.xu))
            # px = np.array(px)
            self.pop.append(px)
        flg=False
        for p in self.pop:
            loss = objF(p , self.wh , self.nets,flg)
            self.ranked_pop.append((loss[0] , p, loss[1], loss[2]))
        self.ranked_pop.sort()
        self.ranked_pop.reverse()

        self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]

    def mutate(self , gene):
        return gene + random.uniform(1-self.mutation_factor , 1+ self.mutation_factor)
    def opt(self , n_iter):
        # ok so take average of last 25 iterations ka loss?
        # yes, isko har iteration pe karne ki jagah har 10 iterstion pe karna hai??
        # Haa kar sakte...
        last_25=0
        # I want to make a queue of 25 elements and keep adding the loss to it and then take the average of it
        
        for iter in range(n_iter):
            if iter % 10 == 0: 
                self.logger.log(f"Iteration {iter} ----> l:{self.ranked_pop[0][0]} h:{self.ranked_pop[0][2]}  o:{self.ranked_pop[0][3]}")
                print(f"Iteration {iter} ----> l:{self.ranked_pop[0][0]} h:{self.ranked_pop[0][2]}  o:{self.ranked_pop[0][3]}")
            self.xHis.append(self.ranked_pop[0])
            # elem = np.array(self.best_pop).T
            # print(elem.shape)
            new_pop = []
            elems = []
            for e in range(self.x_len):
                elem = []
                for j in self.best_pop:
                    elem.append(j[1][e])
                # print(len(elem))
                elems.append(elem)

            for p in range(self.pop_size):
                x = []
                for e in range(self.x_len):
                    # print(len(elems[e]))
                    gene = random.choice(elems[e])
                    gene = gene * random.uniform(1-self.mutation_factor , 1+ self.mutation_factor)
                    x.append(gene)
                new_pop.append(x)

            self.pop = new_pop

            self.ranked_pop = []
            pop_loss=0
            for p in self.pop:
                for i in range(len(p) // 2):
                    p[2*i] = min(p[2*i], self.xu)
                    p[2*i + 1] = min(p[2*i + 1], self.xu)
                loss = self.objF(p , self.wh , self.nets,False)
                pop_loss+=loss[0]
                self.ranked_pop.append((loss[0] , p, loss[1], loss[2]))
            last_25+=pop_loss/self.pop_size 
            if iter % 10 == 0:
                if last_25/25 < pop_loss:
                    break



            self.ranked_pop.sort()
            self.ranked_pop.reverse()
            flg=False
            if(iter==n_iter-1):
                flg=True

            self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]
            loss=self.objF(self.best_pop[0][1],self.wh,self.nets,flg)
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
        self.ga = GA(pop_size , 0.01 , 0.1 , objF , 2*len(self.macros) , 0 ,self.floor.h - max(self.wh) , self.nets , self.wh,1)

    def place(self , iter):
        self.ga.opt(iter)
        X = self.ga.ranked_pop[0][1]
        for i in range(len(self.macros)):
            self.macros[i].x = min(X[(2*i)], self.floor.w - self.wh[2*i])
            self.macros[i].y = min(X[(2*i) + 1], self.floor.h - self.wh[2*i + 1])
        
        # Total loss, HPWL, Overlapping Area
        return self.ga.ranked_pop[0][0], self.ga.ranked_pop[0][2], self.ga.ranked_pop[0][3]

    def genVid(self ,path, full_video=False):
        if not full_video: 
            # generate only an opencv image of the last frame
            # print("Floor Size: ", self.floor.w, self.floor.h)
            img1 = np.zeros((self.floor.h,self.floor.w, 3), np.uint8)
            X = self.ga.xHis[-1][1]
            for i in range(int(len(X) / 2)):
                xi_min = self.macros[i].x
                yi_min = self.macros[i].y
                xi_max = xi_min + self.wh[2*i]
                yi_max = yi_min + self.wh[2*i + 1]
                cv2.rectangle(img1, (int(xi_min) , int(yi_min)) , (int(xi_max) , int(yi_max)) , (0,0,255) , 3)
                cv2.putText(img1, f"{self.macros[i].name}", (int(xi_min), int(yi_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            # print("Shape: ", img1.shape)
            cv2.imwrite(path, img1)

        else:
            out = cv2.VideoWriter(path,0,40, (self.floor.w,self.floor.h))
            for frameX in self.ga.xHis:
                img1 = np.zeros((self.floor.w,self.floor.h, 3), np.uint8)
                X = frameX[1]
                for i in range(int(len(X) / 2)):
                    # print(i)
                    xi_min = X[2*i]
                    yi_min = X[2*i + 1]
                    xi_max = xi_min + self.wh[2*i]
                    yi_max = yi_min + self.wh[2*i + 1]
                    # print(xi_max)
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
        
# https://ieeexplore.ieee.org/document/7033338


def hpwlFaster(X, wh, nets,flg):
    X = np.array(X).reshape(-1, 2)
    s = 0
    for net in nets:
        net_coords = X[net]
        xmin, ymin = net_coords.min(axis=0)
        xmax, ymax = net_coords.max(axis=0)
        s += xmax - xmin + ymax - ymin
    return s


# [[2, 3]
#  [4, 4]
#  [5, 7]]

def getBoundingBox(X, wh):
    X = np.array(X).reshape(-1, 2)
    wh = np.array(wh).reshape(-1, 2)
    x_min, y_min = X[:, 0].min(), X[:, 1].min()
    x_max, y_max = (X + wh)[:, 0].max(), (X + wh)[:, 1].max()
    tot_area = (wh[:, 0] * wh[:, 1]).sum()
    return x_min, y_min, x_max, y_max, tot_area

def isOverlapping(X , wh):
    # print("X: ", X)
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

                if (dx >= 0) and (dy >= 0):
                    logger = Logger.getInstance()
                    logger.log("Macro " + str(i) + " and Macro " + str(j)  + " are overlapping by " + str(dx*dy))
                        # print("Macro ", i , " and Macro ", j , " are overlapping by ", dx*dy)
                    total_overlapp += dx*dy
    
    return total_overlapp

def isOverlappingFaster(X, wh):
    # print("X: ", X)
    # print("wh: ", wh)
    # quit the program

    X = np.array(X).reshape(-1, 2)
    wh = np.array(wh).reshape(-1, 2)
    X_max = X + wh

    dx = np.maximum(0, np.minimum(X_max[:, None, 0], X_max[None, :, 0]) - np.maximum(X[:, None, 0], X[None, :, 0]))
    dy = np.maximum(0, np.minimum(X_max[:, None, 1], X_max[None, :, 1]) - np.maximum(X[:, None, 1], X[None, :, 1]))

    overlap_areas = dx * dy
    # print("Macro ", i , " and Macro ", j , " are overlapping by ", dx*dy)

    np.fill_diagonal(overlap_areas, 0)  # Exclude self-overlap

    total_overlap = np.sum(overlap_areas)
    return total_overlap


def minDist(X , wh):
    mind = math.inf
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

                d = rect_distance(xi_min , yi_min , xi_max , yi_max , xj_min , yj_min , xj_max , yj_max)

                if (d < mind):
                    mind = d
    
    return mind

def has_space(X , wh , threshold):
    md = minDist(X , wh)

    if (md <= 0) or (md > threshold):
        return 1000
    else:
        return 0


                


def rect_distance( x1, y1, x1b, y1b , x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.


def objF(X , wh , nets,flg):
    overlapp=isOverlappingFaster(X , wh)
    hpwl_val = hpwlFaster(X , wh , nets,flg)
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
