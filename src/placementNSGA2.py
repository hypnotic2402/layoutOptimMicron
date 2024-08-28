import numpy as np
import random
import math
import cv2
import copy
from logger import Logger
import cProfile

class Individual:
    def __init__(self, position, objectives):
        self.position = position
        self.objectives = objectives
        self.rank = None
        self.crowding_distance = 0

class NSGAII:
    def __init__(self, pop_size, crossover_prob, mutation_prob, objF, x_len, xl, xu, nets, wh, margin=0):
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.nets = nets
        self.wh = [x+margin for x in wh]
        self.logger = Logger.getInstance()
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            position = [random.uniform(self.xl, self.xu) for _ in range(self.x_len)]
            objectives = self.objF(position, self.wh, self.nets)
            population.append(Individual(position, objectives))
        return population

    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        if random.random() < self.crossover_prob:
            crossover_point = random.randint(1, self.x_len - 1)
            child1.position[crossover_point:], child2.position[crossover_point:] = parent2.position[crossover_point:], parent1.position[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        for i in range(self.x_len):
            if random.random() < self.mutation_prob:
                individual.position[i] = random.uniform(self.xl, self.xu)
        individual.objectives = self.objF(individual.position, self.wh, self.nets)
        # print(individual.objectives)

    def select_parents(self):
        tournament_size = 2
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
            parents.append(tournament[0])
        return parents

    def fast_nondominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.dominated_set = []
            p.domination_count = 0
            for q in population:
                if self.dominates(p, q):
                    p.dominated_set.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def crowding_distance_assignment(self, front):
        if len(front) > 0:
            for p in front:
                p.crowding_distance = 0
            for i in range(len(front[0].objectives)):
                front.sort(key=lambda x: x.objectives[i])
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                max_obj = max(ind.objectives[i] for ind in front)
                min_obj = min(ind.objectives[i] for ind in front)
                if max_obj != min_obj:  # Check to avoid division by zero
                    for j in range(1, len(front) - 1):
                        front[j].crowding_distance += (front[j + 1].objectives[i] - front[j - 1].objectives[i]) / (max_obj - min_obj)


    def dominates(self, individual1, individual2):
        better_in_all = all(x <= y for x, y in zip(individual1.objectives, individual2.objectives))
        better_in_at_least_one = any(x < y for x, y in zip(individual1.objectives, individual2.objectives))
        return better_in_all and better_in_at_least_one

    def generate_offspring(self):
        offspring = []
        while len(offspring) < self.pop_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)
            self.mutate(child1)
            self.mutate(child2)
            offspring.extend([child1, child2])
        return offspring

    def replace_population(self, new_population):
        combined_population = self.population + new_population
        fronts = self.fast_nondominated_sort(combined_population)
        new_population = []
        for front in fronts:
            self.crowding_distance_assignment(front)
            front.sort(key=lambda x: (x.rank, -x.crowding_distance))
            new_population.extend(front)
            if len(new_population) >= self.pop_size:
                break
        self.population = new_population[:self.pop_size]

    def opt(self, n_iter):
        print("n_iter: ", n_iter)
        for iter in range(n_iter):
            if iter % 10 == 0:
                best = min(self.population, key=lambda x: x.objectives[0])
                # best = self.population[-1]
                # print(self.population[0])
                self.logger.log(f"Iteration {iter} ----> Overlap: {best.objectives[0]} HPWL: {best.objectives[1]} Distance: {best.objectives[2]}")
                print(f"Iteration {iter} ----> Overlap: {best.objectives[0]} HPWL: {best.objectives[1]} Distance: {best.objectives[2]}")
            offspring = self.generate_offspring()
            self.replace_population(offspring)

        self.best = min(self.population, key=lambda x: x.objectives[0])
        isOverlapping(self.best.position, self.wh)
        x_min, y_min, x_max, y_max, tot_area = getBoundingBox(self.best.position, self.wh)

class PlacementSolver:
    floor = None
    def __init__(self, macros, nets, floor, pop_size, margin=0):
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
        self.nsga2 = NSGAII(pop_size, 0.7, 0.01, objF, 2*len(self.macros), 0, self.floor.h - max(self.wh), self.nets, self.wh, 1)

    def place(self, iter):
        print("NSGA2 in here...")
        self.nsga2.opt(iter)
        X = self.nsga2.best.position
        for i in range(len(self.macros)):
            self.macros[i].x = min(X[(2*i)], self.floor.w - self.wh[2*i])
            self.macros[i].y = min(X[(2*i) + 1], self.floor.h - self.wh[2*i + 1])

    def genVid(self, path, full_video=False):
        if not full_video:
            img1 = np.zeros((self.floor.h, self.floor.w, 3), np.uint8)
            X = self.nsga2.population[0].position
            for i in range(int(len(X) / 2)):
                xi_min = self.macros[i].x
                yi_min = self.macros[i].y
                xi_max = xi_min + self.wh[2*i]
                yi_max = yi_min + self.wh[2*i + 1]
                cv2.rectangle(img1, (int(xi_min), int(yi_min)), (int(xi_max), int(yi_max)), (0,0,255), 3)
                cv2.putText(img1, f"{self.macros[i].name}", (int(xi_min), int(yi_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imwrite(path, img1)
        else:
            out = cv2.VideoWriter(path, 0, 40, (self.floor.w, self.floor.h))
            for frame in self.nsga2.population:
                img1 = np.zeros((self.floor.w, self.floor.h, 3), np.uint8)
                X = frame.position
                for i in range(int(len(X) / 2)):
                    xi_min = X[2*i]
                    yi_min = X[2*i + 1]
                    xi_max = xi_min + self.wh[2*i]
                    yi_max = yi_min + self.wh[2*i + 1]
                    cv2.rectangle(img1, (int(xi_min), int(yi_min)), (int(xi_max), int(yi_max)), (0, 0, 255), 3)
                out.write(img1)
            cv2.destroyAllWindows()
            out.release()

def hpwl(X, wh, nets):
    s = 0
    for net in nets:
        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf
        for i in range(len(net)):
            x = X[2*net[i]]
            y = X[2*net[i] + 1]
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        s += (xmax - xmin) + (ymax - ymin)
    return s

def hpwlFaster(X, wh, nets):
    s = 0
    for net in nets:
        xmin = min(X[2*net[i]] for i in range(len(net)))
        xmax = max(X[2*net[i]] for i in range(len(net)))
        ymin = min(X[2*net[i] + 1] for i in range(len(net)))
        ymax = max(X[2*net[i] + 1] for i in range(len(net)))
        s += (xmax - xmin) + (ymax - ymin)
    return s

def getBoundingBox(X, wh):
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    total_area = 0
    for i in range(int(len(X) / 2)):
        xi_min = X[2*i]
        yi_min = X[2*i + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        x_min = min(x_min, xi_min)
        x_max = max(x_max, xi_max)
        y_min = min(y_min, yi_min)
        y_max = max(y_max, yi_max)
        total_area += (yi_max - yi_min) * (xi_max - xi_min)
    return x_min, y_min, x_max, y_max, total_area

def distance_from_edges(X, wh, floor_w, floor_h):
    distance = 0
    for i in range(int(len(X) / 2)):
        xi_min = X[2*i]
        yi_min = X[2*i + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        distance += xi_min + yi_min + (floor_w - xi_max) + (floor_h - yi_max)
    return distance

def isOverlapping(X, wh):
    total_overlapp = 0
    for i in range(int(len(X) / 2)):
        xi_min = X[2*i]
        yi_min = X[2*i + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        for j in range(int(len(X) / 2)):
            if i != j:
                xj_min = X[2*j]
                yj_min = X[2*j + 1]
                xj_max = xj_min + wh[2*j]
                yj_max = yj_min + wh[2*j + 1]
                dx = min(xi_max , xj_max) - max(xi_min , xj_min)
                dy = min(yi_max , yj_max) - max(yi_min , yj_min)
                if (dx>=0) and (dy>=0):
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

    np.fill_diagonal(overlap_areas, 0)  # Exclude self-overlap

    total_overlap = np.sum(overlap_areas)
    return total_overlap

def objF(X , wh , nets):
    overlap = isOverlappingFaster(X, wh)
    _, _, _, _, total_area = getBoundingBox(X, wh)
    floor_w, floor_h = PlacementSolver.floor.w, PlacementSolver.floor.h
    distance = distance_from_edges(X, wh, floor_w, floor_h)
    hpwl = hpwlFaster(X, wh, nets)
    return overlap, hpwl, distance

# 
