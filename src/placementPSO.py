import numpy as np
import random
import math
import classes as cls
import cv2
import time
from logger import Logger

class Particle:
    def __init__(self , x_len , xl , xu):
        self.position = np.array([random.uniform(xl, xu) for _ in range(x_len)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(x_len)])
        self.best_position = np.copy(self.position)
        self.best_score = -np.inf

class PSO:
    def __init__(self, pop_size, inertia, cognitive_coeff, social_coeff, objF, x_len, xl, xu, nets, wh, margin=0):
        self.pop_size = pop_size
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.nets = nets
        self.wh = [x + margin for x in wh]
        self.logger = Logger.getInstance()
        self.particles = [Particle(x_len, xl, xu) for _ in range(self.pop_size)]
        self.global_best_position = np.zeros(x_len)
        self.global_best_score = -np.inf
        self.xHis = []

        for particle in self.particles:
            flg = False
            score, hpwl_val, overlapp = self.objF(particle.position, self.wh, self.nets, flg)
            particle.best_score = score
            particle.best_position = np.copy(particle.position)
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = np.copy(particle.position)

    def optimize(self, n_iter):
        last_25 = 0
        
        for iter in range(n_iter):
                
            self.xHis.append((self.global_best_score, self.global_best_position))

            for particle in self.particles:
                r1 = random.random()
                r2 = random.random()

                cognitive_velocity = self.cognitive_coeff * r1 * (particle.best_position - particle.position)
                social_velocity = self.social_coeff * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.inertia * particle.velocity + cognitive_velocity + social_velocity
                particle.position += particle.velocity

                particle.position = np.clip(particle.position, self.xl, self.xu)

                flg = False
                score, hpwl_val, overlapp = self.objF(particle.position, self.wh, self.nets, flg)
                
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)
                
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(particle.position)
            
            if iter % 10 == 0: 
                overlapp = self.objF(self.global_best_position, self.wh, self.nets, False)[2]
                hpwl_val = self.objF(self.global_best_position, self.wh, self.nets, False)[1]
                # self.logger.log(f"Iteration {iter} ----> Global Best: {self.global_best_score}, HPWL: {hpwl_val}, Overlapp: {overlapp}")
                print(f"Iteration {iter} ----> Global Best: {self.global_best_score}, HPWL: {hpwl_val}, Overlapp: {overlapp}")
            
        
        isOverlappingFaster(self.global_best_position, self.wh)
        x_min, y_min, x_max, y_max, tot_area = getBoundingBox(self.global_best_position, self.wh)

class PlacementSolver:
    floor = None

    def __init__(self, macros, nets, floor, pop_size, margin=0):
        self.macros = macros
        self.Nets = nets
        self.floor = floor
        PlacementSolver.floor = floor
        self.nets = [[x.id for x in net.macros] for net in nets]
        self.wh = [dim for macro in macros for dim in (macro.w, macro.h)]
        self.pso = PSO(pop_size, inertia=0.7, cognitive_coeff=1.5, social_coeff=1.5, 
                       objF=objF, x_len=2*len(self.macros), xl=0, xu=floor.h - max(self.wh), 
                       nets=self.nets, wh=self.wh, margin=margin)

    def place(self, iter):
        self.pso.optimize(iter)
        X = self.pso.global_best_position
        for i in range(len(self.macros)):
            self.macros[i].x = min(X[2*i], self.floor.w - self.wh[2*i])
            self.macros[i].y = min(X[2*i + 1], self.floor.h - self.wh[2*i + 1])

        return self.pso.global_best_score, hpwlFaster(self.pso.global_best_position, self.wh, self.nets), \
               isOverlappingFaster(self.pso.global_best_position, self.wh)

    def genVid(self, path, full_video=False):
        if not full_video: 
            img1 = np.zeros((self.floor.h, self.floor.w, 3), np.uint8)
            X = self.pso.xHis[-1][1]
            for i in range(int(len(X) / 2)):
                xi_min = self.macros[i].x
                yi_min = self.macros[i].y
                xi_max = xi_min + self.wh[2*i]
                yi_max = yi_min + self.wh[2*i + 1]
                cv2.rectangle(img1, (int(xi_min), int(yi_min)), (int(xi_max), int(yi_max)), (0, 0, 255), 3)
                cv2.putText(img1, f"{self.macros[i].name}", (int(xi_min), int(yi_min) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(path, img1)
        else:
            out = cv2.VideoWriter(path, 0, 40, (self.floor.w, self.floor.h))
            for frameX in self.pso.xHis:
                img1 = np.zeros((self.floor.w, self.floor.h, 3), np.uint8)
                X = frameX[1]
                for i in range(int(len(X) / 2)):
                    xi_min = X[2*i]
                    yi_min = X[2*i + 1]
                    xi_max = xi_min + self.wh[2*i]
                    yi_max = yi_min + self.wh[2*i + 1]
                    cv2.rectangle(img1, (int(xi_min), int(yi_min)), (int(xi_max), int(yi_max)), (0, 0, 255), 3)
                out.write(img1)
            cv2.destroyAllWindows()
            out.release()


def hpwlFaster(X, wh, nets):
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
                    # logger.log("Macro " + str(i) + " and Macro " + str(j)  + " are overlapping by " + str(dx*dy))
                    total_overlapp += dx*dy
    
    return total_overlapp

def isOverlappingFaster(X, wh):
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

    return (- 1 * hpwl_val - (len(X) ** 2) * overlapp , hpwl_val, overlapp)
