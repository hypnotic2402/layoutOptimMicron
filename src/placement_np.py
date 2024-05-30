import numpy as np
import random
import math
import classes as cls
import cv2
import time

class Individual:
    def __init__(self , xu , xl , yu , yl):
        self.x = random.uniform(xl, xu)
        self.y = random.uniform(yl, yu)
        

class GA: 
    def __init__(self , pop_size , mutation_factor, crossover_factor , objF , x_len , xl, xu , nets , wh ,margin=0):
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_factor = crossover_factor
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.wh = np.array(wh) + margin
        self.nets = nets
        self.xHis = []

        # Generate population
        self.pop = np.random.uniform(self.xl, self.xu, (self.pop_size, self.x_len))

        # Calculate loss for each individual in the population
        self.ranked_pop = [(self.objF(p, self.wh, self.nets)[0], p, self.objF(p, self.wh, self.nets)[1], self.objF(p, self.wh, self.nets)[2]) for p in self.pop]
        self.ranked_pop.sort(reverse=True)

        self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]

    def mutate(self, genes):
        return genes + np.random.uniform(1 - self.mutation_factor, 1 + self.mutation_factor, size=genes.shape)

    def opt(self, n_iter):
        for iter in range(n_iter):
            if iter % 10 == 0: 
                print(f"Iteration {iter} ----> l:{self.ranked_pop[0][0]} h:{self.ranked_pop[0][2]}  o:{self.ranked_pop[0][3]}")
            self.xHis.append(self.ranked_pop[0])

            # Create a 2D array where each row contains the e-th gene of the best individuals
            elems = np.array([individual[1] for individual in self.best_pop]).T

            # Generate new population
            new_pop = np.array([np.random.choice(elems[e], self.pop_size) for e in range(self.x_len)]).T
            new_pop *= np.random.uniform(1 - self.mutation_factor, 1 + self.mutation_factor, size=new_pop.shape)

            # Clip the genes to the upper bound
            new_pop = np.minimum(new_pop, self.xu)

            # Calculate loss for each individual in the new population
            self.ranked_pop = [(self.objF(p, self.wh, self.nets)[0], p, self.objF(p, self.wh, self.nets)[1], self.objF(p, self.wh, self.nets)[2]) for p in new_pop]
            self.ranked_pop.sort(reverse=True)

            self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]
            self.pop = new_pop
        

class PlacementSolver:
    def __init__(self, macros, nets, floor, pop_size, margin=0):
        self.macros = macros
        self.Nets = nets
        self.floor = floor
        self.margin = margin

        # Use list comprehension to create self.nets
        self.nets = [[x.id for x in net.macros] for net in self.Nets]

        # Use list comprehension and numpy to create self.wh
        self.wh = np.array([[macro.w, macro.h] for macro in macros]).flatten().tolist()

        self.ga = GA(pop_size, 0.1, 0.1, objF, 2*len(self.macros), 0, self.floor.h - max(self.wh), self.nets, self.wh)

    def place(self, iter):
        self.ga.opt(iter)
        X = np.array(self.ga.xHis[-1][1])

        x_coords = np.minimum(X[::2], self.floor.w - np.array(self.wh[::2]))
        y_coords = np.minimum(X[1::2], self.floor.h - np.array(self.wh[1::2]))

        for i, macro in enumerate(self.macros):
            macro.x = x_coords[i]
            macro.y = y_coords[i]


    def genVid(self, path, full_video=False):
        if not full_video: 
            # generate only an opencv image of the last frame
            print("Floor Size: ", self.floor.w, self.floor.h)
            img1 = np.zeros((self.floor.w, self.floor.h, 3), np.uint8)
            X = np.array(self.ga.xHis[-1][1])

            # Create arrays for the x and y coordinates and the widths and heights of the macros
            x_coords = X[::2]
            y_coords = X[1::2]
            widths = np.array(self.wh[::2])
            heights = np.array(self.wh[1::2])

            # Calculate the maximum x and y coordinates
            x_max_coords = x_coords + widths
            y_max_coords = y_coords + heights

            # Draw the rectangles and the text
            for i, macro in enumerate(self.macros):
                cv2.rectangle(img1, (int(x_coords[i]), int(y_coords[i])), (int(x_max_coords[i]), int(y_max_coords[i])), (0,0,255), 3)
                cv2.putText(img1, f"{macro.name}", (int(x_coords[i]), int(y_coords[i])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imwrite(path, img1)

        else:
            out = cv2.VideoWriter(path,0,40, (self.floor.w,self.floor.h))
            for frameX in self.ga.xHis:
                img1 = np.zeros((self.floor.w,self.floor.h, 3), np.uint8)
                X = np.array(frameX[1])

                # Create arrays for the x and y coordinates and the widths and heights of the macros
                x_coords = X[::2]
                y_coords = X[1::2]
                widths = np.array(self.wh[::2])
                heights = np.array(self.wh[1::2])

                # Calculate the maximum x and y coordinates
                x_max_coords = x_coords + widths
                y_max_coords = y_coords + heights

                # Draw the rectangles
                for i in range(len(x_coords)):
                    cv2.rectangle(img1, (int(x_coords[i]), int(y_coords[i])), (int(x_max_coords[i]), int(y_max_coords[i])), (0,0,255), 3)

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


def hpwlFaster(X, wh, nets):
    X = np.array(X).reshape(-1, 2)
    s = 0
    for net in nets:
        net_coords = X[net]
        xmin, ymin = net_coords.min(axis=0)
        xmax, ymax = net_coords.max(axis=0)
        s += xmax - xmin + ymax - ymin
    return s




def isOverlapping(X , wh):
    total_overlapp = 0
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
                    total_overlapp += dx*dy
    
    return total_overlapp

def isOverlappingFaster(X, wh):
    X = np.array(X).reshape(-1, 2)
    wh = np.array(wh).reshape(-1, 2)
    X_max = X + wh

    dx = np.maximum(0, np.minimum(X_max[:, None, 0], X_max[None, :, 0]) - np.maximum(X[:, None, 0], X[None, :, 0]))
    dy = np.maximum(0, np.minimum(X_max[:, None, 1], X_max[None, :, 1]) - np.maximum(X[:, None, 1], X[None, :, 1]))

    overlap_areas = dx * dy
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


def objF(X , wh , nets):
    overlapp=isOverlappingFaster(X , wh)
    hpwl_val = hpwlFaster(X , wh , nets)
    # print("Overlapping: ", overlapp, " HPWL: ", hpwl_val)
    # return (- 1/len(X) * hpwl_val - 10000*overlapp*len(X), hpwl_val, overlapp)
    return (- 1 * hpwl_val - 2.5 * overlapp, hpwl_val, overlapp)