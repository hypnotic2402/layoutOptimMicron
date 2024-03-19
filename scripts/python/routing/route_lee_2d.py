import numpy as np
import pandas as pd
from collections import deque  
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap


class Cell:
    def __init__(self , x , y):
        self.x = x
        self.y = y
        # self.mat = 

class queueNode:  
    def __init__(self,point: Cell, dist: int):  
        self.point = point # Cell coordinates  
        self.dist = dist # Cell's distance from the source  

class Net:
    def __init__(self , src , dst):
        self.src = src
        self.dst = dst
        self.routed = []
        

class RoutingSolver:

    def __init__(self , dim_x , dim_y):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.visited = [[0 for _ in range(dim_x)] for _ in range(dim_y)] 
        self.rowNum = [-1, 0, 0, 1]  
        self.colNum = [0, -1, 1, 0]  
        self.matr = [[1 for _ in range(dim_x)] for _ in range(dim_y)]  # 0 is blocked, 1 is available
        self.distMat = [[0 for _ in range(dim_x)] for _ in range(dim_y)] 

    def check_valid(self , row , col):
        return ((row >= 0) and (row < self.dim_x) and (col >= 0) and (col < self.dim_y))
    
    def LeeAlgo(self , mat , src:Cell, dest:Cell):

        if (mat[src.x][src.y]!=1 or mat[dest.x][dest.y]) != 1:
            return -1
        
        self.visited = [[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] 

        self.visited[src.x][src.y] = 1

        q = deque()

        s = queueNode(src,0)  
        q.append(s)

        while q:

            curr = q.popleft()
            point = curr.point

            if (point.x == dest.x) and (point.y == dest.y):
                # self.distMat[point.x][point.y] = -1
                return curr.dist
            
            for i in range(4):
                row = point.x + self.rowNum[i]
                col = point.y + self.colNum[i]

                if (self.check_valid(row , col)) and (mat[row][col] == 1) and (self.visited[row][col] == 0):
                    self.visited[row][col] = 1
                    self.distMat[row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col),  curr.dist+1)
                    q.append(Adjcell)

        return -1  
    
    def check_path(self , src , dst):
        print(self.LeeAlgo(self.matr , src , dst))

    def findNextNode(self , distMat , curr_node , curr_dir):
        dir = curr_dir
        currDist = distMat[curr_node.x][curr_node.y]
        # nextX = curr_node.x + self.rowNum[dir]
        # nextY = curr_node.y + self.colNum[dir]

        for i in range(4):
            # print(dir)
            nextX = curr_node.x + self.rowNum[dir]
            nextY = curr_node.y + self.colNum[dir]

            if (distMat[nextX][nextY] == currDist - 1):
                return nextX , nextY , dir

            if (dir == 3): dir = 0
            else: dir += 1

        return -1

        


    def computeNet(self , src , dst):

        dist = self.LeeAlgo(self.matr , src , dst)

        curr_dir = 0
        curr_node = dst
        wires = []

        while(self.distMat[curr_node.x][curr_node.y] != 0):

            self.matr[curr_node.x][curr_node.y] = 0
            wires.append(curr_node)
            nextX , nextY , curr_dir = self.findNextNode(self.distMat , curr_node , curr_dir)
            curr_node = Cell(nextX , nextY)

        self.matr[curr_node.x][curr_node.y] = 0
        wires.append(curr_node)

        self.distMat = [[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] 
        return wires

    def route(self , nets):
        i = 1
        for net in nets:
            print("Routing Net " + str(i))
            net.routed = self.computeNet(net.src , net.dst) 
            i+=1

    def display_curr_matr(self , nets , plot):
        mat = [[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)]
        # print(mat)
        for line in mat:
            print(line)
        print("")

        if plot:
            plt.imshow(np.array(mat), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.show()
                
        i = 1
        for net in nets:
            for cell in net.routed:
                mat[cell.x][cell.y] = i

            i += 1

            for line in mat:
                print(line)
            print("")

            if plot:
                plt.imshow(np.array(mat), cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.show()

        return mat


            



        
        
if __name__== "__main__":
    RS = RoutingSolver(5,5)
    nets = [Net(Cell(2,1) , Cell(4,4)) , Net(Cell(0,3) , Cell(4,3))]
    RS.route(nets)
    x = RS.display_curr_matr(nets , 0)