import numpy as np
import pandas as pd
from collections import deque  
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap

class Cell:
    def __init__(self , x , y , z):
        self.x = x
        self.y = y
        self.z = z

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

    def __init__(self , dim_x , dim_y , layers):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = layers
        self.visited = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        # self.rowNum = [-1, 0, 0, 1]  
        # self.colNum = [0, -1, 1, 0]
        self.rowNume =  [1 , -1 , 0 , 0]
        self.colNume =  [0 , 0 , 0 , 0]
        self.layerNume =[0 , 0 , 1 , -1] 
        self.rowNumo =  [0 , 0 , 0 , 0]
        self.colNumo =  [1 , -1 , 0 , 0]
        self.layerNumo =[0 , 0 , 1 , -1] 

        self.matr = [[[1 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        self.distMat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]

    def check_valid(self , row , col , layer):
        return ((row >= 0) and (row < self.dim_x) and (col >= 0) and (col < self.dim_y) and (layer < self.dim_z) and (layer >= 0))


    def LeeAlgo(self , mat , src:Cell, dest:Cell):

        if (mat[src.z][src.x][src.y]!=1 or mat[dest.z][dest.x][dest.y]) != 1:
            return -1

        self.visited = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]

        self.visited[src.z][src.x][src.y] = 1

        q = deque()

        s = queueNode(src,0)  
        q.append(s)

        while q:

            curr = q.popleft()
            point = curr.point

            if (point.x == dest.x) and (point.y == dest.y) and (point.z == dest.z):
                return curr.dist

            if (point.z % 2 == 0):
                
                row = point.x + 1
                col = point.y
                layr = point.z

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x - 1
                col = point.y
                layr = point.z

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x 
                col = point.y
                layr = point.z + 1

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x 
                col = point.y
                layr = point.z - 1

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

            else:

                row = point.x 
                col = point.y + 1
                layr = point.z

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x
                col = point.y - 1
                layr = point.z

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x 
                col = point.y
                layr = point.z + 1

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)

                row = point.x 
                col = point.y
                layr = point.z - 1

                if (self.check_valid(row , col , layr)) and (mat[layr][row][col] == 1) and (self.visited[layr][row][col] == 0):
                    self.visited[layr][row][col] = 1
                    self.distMat[layr][row][col] = curr.dist+1
                    Adjcell = queueNode(Cell(row,col,layr),  curr.dist+1)
                    q.append(Adjcell)


        return -1

    def check_path(self , src , dst):
        print(self.LeeAlgo(self.matr , src , dst))

    def findNextNode(self , distMat , curr_node , curr_dir):
        dir = curr_dir
        currDist = distMat[curr_node.z][curr_node.x][curr_node.y]
        # xDir = []

        if (curr_node.z % 2 == 0):

            nextX = curr_node.x + dir
            nextY = curr_node.y
            nextZ = curr_node.z

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir
                
            dir = -dir

            nextX = curr_node.x + dir
            nextY = curr_node.y
            nextZ = curr_node.z

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

            nextX = curr_node.x 
            nextY = curr_node.y
            nextZ = curr_node.z + dir

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

            dir = -dir

            nextX = curr_node.x 
            nextY = curr_node.y
            nextZ = curr_node.z + dir

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

        else:

            nextX = curr_node.x 
            nextY = curr_node.y + dir
            nextZ = curr_node.z

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir
                
            dir = -dir

            nextX = curr_node.x 
            nextY = curr_node.y + dir
            nextZ = curr_node.z

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

            nextX = curr_node.x 
            nextY = curr_node.y
            nextZ = curr_node.z + dir

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

            dir = -dir

            nextX = curr_node.x 
            nextY = curr_node.y
            nextZ = curr_node.z + dir

            if (self.check_valid(nextX , nextY , nextZ)):
                if (distMat[nextZ][nextX][nextY] == currDist - 1):
                    return nextX , nextY , nextZ , dir

        return -1
            
    def computeNet(self , src , dst):

        dist = self.LeeAlgo(self.matr , src , dst)
        if (dist == -1):
            print("Path not found")

        curr_dir = 1
        curr_node = dst
        wires = []

        while(self.distMat[curr_node.z][curr_node.x][curr_node.y] != 0):
            self.matr[curr_node.z][curr_node.x][curr_node.y] = 0
            wires.append(curr_node)
            nextX , nextY , nextZ , curr_dir = self.findNextNode(self.distMat , curr_node , curr_dir)
            curr_node = Cell(nextX , nextY , nextZ)

        self.matr[curr_node.z][curr_node.y][curr_node.z] = 0
        wires.append(curr_node)
        self.distMat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        return wires

    def route(self , nets):
        i = 1

        for net in nets:
            self.matr[net.src.z][net.src.x][net.src.y] = 0
            self.matr[net.dst.z][net.dst.x][net.dst.y] = 0

        for net in nets:
            print("Routing Net " + str(i))
            self.matr[net.src.z][net.src.x][net.src.y] = 1
            self.matr[net.dst.z][net.dst.x][net.dst.y] = 1
            net.routed = self.computeNet(net.src , net.dst) 
            i+=1


    def display_curr_matr(self , nets , plot):
        mat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        l = 0
        for layer in mat:
            print("Layer " + str(l))
            for line in layer:
                print(line)

            print("")
            l+=1
        print("--------------------------------------------")
        if plot == 2:
            j = 1
            mat2 = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
            for net in nets:
                mat2[net.src.z][net.src.x][net.src.y] = j
                mat2[net.dst.z][net.dst.x][net.dst.y] = j
                j+=1

            plt.imshow(np.array(mat2[0]), cmap='viridis', interpolation='nearest')
            plt.show()
            # for sp in range(self.dim_z):
            #     plt.imshow(np.array(mat2[sp]), cmap='viridis', interpolation='nearest')
            #     plt.show()

        if plot == 1:
            plt.imshow(np.array(mat[0]), cmap='viridis', interpolation='nearest')

        i = 1

        for net in nets:
            for cell in net.routed:
                mat[cell.z][cell.x][cell.y] = i

            i += 1
            l = 0
            for layer in mat:
                print("Layer " + str(l))
                for line in layer:
                    print(line)

                print("")
                l+=1
            print("--------------------------------------------")

            if (plot == 1):
                for sp in range(self.dim_z):
                    plt.title("Layer " + str(sp))
                    plt.imshow(np.array(mat[sp]), cmap='viridis', interpolation='nearest')
                    plt.show()

        if (plot == 2):
            for sp in range(self.dim_z):
                plt.title("Layer " + str(sp))
                plt.imshow(np.array(mat[sp]), cmap='viridis', interpolation='nearest')
                plt.show()

                

        
          



if __name__== "__main__":
    RS = RoutingSolver(20,20,2)
    nets = [Net(Cell(0,10,0) , Cell(16,11,0)) , Net(Cell(2,1,0) , Cell(18,18,0)) ,  Net(Cell(1,1,0) , Cell(10,3,0))]
    # nets = [Net(Cell(0,0,0) , Cell(3,3,0)) , Net(Cell(2,0,0) , Cell(2,4,0))]
    RS.route(nets)
    RS.display_curr_matr(nets , 2)