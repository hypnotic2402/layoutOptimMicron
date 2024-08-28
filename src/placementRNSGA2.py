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

class RNSGAII:
    def __init__(self, pop_size, crossover_prob, mutation_prob, objF, x_len, xl, xu, nets, wh, ref_point, margin=0):
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objF = objF
        self.x_len = x_len
        self.xl = xl
        self.xu = xu
        self.nets = nets
        self.wh = [x + margin for x in wh]
        self.ref_point = ref_point
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
            front.sort(key=lambda x: self.calculate_r2_indicator(x))
            new_population.extend(front)
            if len(new_population) >= self.pop_size:
                break
        self.population = new_population[:self.pop_size]

    def calculate_r2_indicator(self, individual):
        r2_value = max([w * abs(o - rp) for w, o, rp in zip([1]*len(individual.objectives), individual.objectives, self.ref_point)])
        return r2_value

    def opt(self, n_iter):
        print("n_iter: ", n_iter)
        for iter in range(n_iter):
            if iter % 10 == 0:
                best = min(self.population, key=lambda x: self.calculate_r2_indicator(x))
                self.logger.log(f"Iteration {iter} ----> Overlap: {best.objectives[0]} HPWL: {best.objectives[1]} Distance: {best.objectives[2]}")
                print(f"Iteration {iter} ----> Overlap: {best.objectives[0]} HPWL: {best.objectives[1]} Distance: {best.objectives[2]}")
            offspring = self.generate_offspring()
            self.replace_population(offspring)

        self.best = min(self.population, key=lambda x: self.calculate_r2_indicator(x))
        isOverlapping(self.best.position, self.wh)
        x_min, y_min, x_max, y_max, tot_area = getBoundingBox(self.best.position, self.wh)

class PlacementSolver:
    floor = None
    def __init__(self, macros, nets, floor, pop_size, ref_point, margin=0):
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
        self.rnsga2 = RNSGAII(pop_size, 0.7, 0.01, objF, 2*len(self.macros), 0, self.floor.h - max(self.wh), self.nets, self.wh, ref_point, 1)
    def place(self, iter):
        print("RNSGA2 in here...")
        self.rnsga2.opt(iter)
        X = self.rnsga2.best.position
        for i in range(len(self.macros)):
            self.macros[i].x = min(X[(2*i)], self.floor.w - self.wh[2*i])
            self.macros[i].y = min(X[(2*i) + 1], self.floor.h - self.wh[2*i + 1])

    def genVid(self, path, full_video=False):
        if not full_video:
            img1 = np.zeros((self.floor.h, self.floor.w, 3), np.uint8)
            X = self.rnsga2.population[0].position
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
            for frame in self.rnsga2.population:
                img1 = np.zeros((self.floor.w, self.floor.h, 3), np.uint8)
                X = frame.position
                for i in range(int(len(X) / 2)):
                    xi_min = X[2*i]
                    yi_min = X[2*i + 1]
                    xi_max = xi_min + self.wh[2*i]
                    yi_max = yi_min + self.wh[2*i + 1]
                    cv2.rectangle(img1, (int(xi_min), int(yi_min)), (int(xi_max), int(yi_max)), (0,0,255), 3)
                    cv2.putText(img1, f"{self.macros[i].name}", (int(xi_min), int(yi_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                out.write(img1)
            out.release()

def isOverlapping(X, wh):
    for i in range(int(len(X) / 2)):
        xi_min = X[(2*i)]
        yi_min = X[(2*i) + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        for j in range(i+1, int(len(X) / 2)):
            xj_min = X[(2*j)]
            yj_min = X[(2*j) + 1]
            xj_max = xj_min + wh[2*j]
            yj_max = yj_min + wh[2*j + 1]
            if (xi_max > xj_min and xj_max > xi_min and yi_max > yj_min and yj_max > yi_min):
                print(f"Overlap between {i+1} and {j+1} with coordinates: {xi_min}, {yi_min}, {xi_max}, {yi_max} and {xj_min}, {yj_min}, {xj_max}, {yj_max}")

def objF(X, wh, nets):
    overlap_area = 0
    hpwl = 0
    min_dist = 0
    for i in range(int(len(X) / 2)):
        xi_min = X[(2*i)]
        yi_min = X[(2*i) + 1]
        xi_max = xi_min + wh[2*i]
        yi_max = yi_min + wh[2*i + 1]
        for j in range(i+1, int(len(X) / 2)):
            xj_min = X[(2*j)]
            yj_min = X[(2*j) + 1]
            xj_max = xj_min + wh[2*j]
            yj_max = yj_min + wh[2*j + 1]
            dx = min(xi_max, xj_max) - max(xi_min, xj_min)
            dy = min(yi_max, yj_max) - max(yi_min, yj_min)
            if (dx >= 0 and dy >= 0):
                overlap_area += dx * dy

    Xx, Xy = [], []
    for i in range(int(len(X) / 2)):
        Xx.append(X[(2*i)] + wh[2*i] / 2)
        Xy.append(X[(2*i) + 1] + wh[2*i + 1] / 2)

    for net in nets:
        x = [Xx[i] for i in net]
        y = [Xy[i] for i in net]
        hpwl += (max(x) - min(x) + max(y) - min(y))

    min_dist = getBoundingBox(X, wh)[4]
    return (overlap_area, hpwl, min_dist)

def getBoundingBox(X, wh):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
    area = 0
    for i in range(int(len(X) / 2)):
        min_x = min(min_x, X[(2*i)])
        min_y = min(min_y, X[(2*i) + 1])
        max_x = max(max_x, X[(2*i)] + wh[2*i])
        max_y = max(max_y, X[(2*i) + 1] + wh[2*i + 1])
        area += wh[2*i] * wh[2*i + 1]
    return min_x, min_y, max_x, max_y, ((max_x - min_x) * (max_y - min_y)) - area


