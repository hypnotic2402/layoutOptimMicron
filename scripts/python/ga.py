import numpy as np
import random
import math

class Individual:
    def __init__(self , xu , xl , yu , yl):
        self.x = random.uniform(xl, xu)
        self.y = random.uniform(yl, yu)
        

class GA: 
    def __init__(self , pop_size , mutation_factor, crossover_factor , objF , x_len , xl, xu , nets , wh):
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
        

        for p in range(self.pop_size):
            px = []
            for x in range(self.x_len):
                px.append(random.uniform(self.xl,self.xu))
            # px = np.array(px)
            self.pop.append(px)

        for p in self.pop:
            self.ranked_pop.append((objF(p , self.wh , self.nets) , p))
        self.ranked_pop.sort()
        self.ranked_pop.reverse()

        self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]

    def mutate(self , gene):
        return gene + random.uniform(1-self.mutation_factor , 1+ self.mutation_factor)

    def opt(self , n_iter):
        for i in range(n_iter):
            print("Iteration " + str(i) + " ----> " + str(self.ranked_pop[0]))
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

            for p in self.pop:
                self.ranked_pop.append((self.objF(p , self.wh , self.nets) , p))
            self.ranked_pop.sort()
            self.ranked_pop.reverse()

            self.best_pop = self.ranked_pop[:int(round(self.pop_size * self.crossover_factor))]

        


        



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


def isOverlapping(X , wh):
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


def objF(X , wh , nets):
    return (- hpwl(X , wh , nets) - 1000*isOverlapping(X , wh))
    

