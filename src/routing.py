import classes as cls
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
    def __init__(self,cells):
        self.cells=cells
        self.routed=[]
        self.hpwl = 0

    def calcHPWL(self):
        self.hpwl = abs(max([c.x for c in self.cells]) - min([c.x for c in self.cells])) + abs(max([c.y for c in self.cells]) - min([c.y for c in self.cells]))

class RoutingSolver:

    def __init__(self, macros , nets , floor):
        self.macros = macros
        self.nets = nets
        self.floor = floor
        self.dim_x = int(self.floor.w / self.floor.gridUnit)
        self.dim_y = int(self.floor.h / self.floor.gridUnit)
        # print(self.dim_x)
        
        self.layers = self.floor.layers
        self.dim_z = self.layers
        self.visited = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        self.Nets = []
        for net in self.nets:
            currNet = Net([])
            for macro in net.macros:
                for pin in macro.pins:
                    xi = macro.x + pin.x
                    yi = macro.y + pin.y
                    xi = int(xi/self.floor.gridUnit)
                    yi = int(yi/self.floor.gridUnit) 

                    

                    currNet.cells.append(Cell(xi,yi,0))

            self.Nets.append(currNet)

        for net in self.Nets:
            net.calcHPWL()

        self.rowNume =  [1 , -1 , 0 , 0]
        self.colNume =  [0 , 0 , 0 , 0]
        self.layerNume =[0 , 0 , 1 , -1] 
        self.rowNumo =  [0 , 0 , 0 , 0]
        self.colNumo =  [1 , -1 , 0 , 0]
        self.layerNumo =[0 , 0 , 1 , -1] 

        self.matr = [[[1 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        self.distMat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]

    def block_cells(self , cells):
        for cell in cells:
            self.matr[cell.z][cell.x][cell.y] = 0

    def check_valid(self , row , col , layer):
        return ((row >= 0) and (row < self.dim_x) and (col >= 0) and (col < self.dim_y) and (layer < self.dim_z) and (layer >= 0))

    def find_consecutive_coords(self,nets):
            consecutive_coords = []
            for net in nets:
                for i in range(len(net.routed) - 1):
                    cell1 = net.routed[i]
                    cell2 = net.routed[i + 1]
                    if cell1.x == cell2.x and cell1.y == cell2.y and cell1.z != cell2.z:
                        consecutive_coords.append((cell1.x, cell2.y))
            return consecutive_coords
    
    def LeeAlgo(self , mat , dest:Cell, net:Net):

        if (mat[dest.z][dest.x][dest.y]) != 1:
            return -1,None
        # for cell in net.routed:
                        # print(cell.x,cell.y,cell.z)
        self.visited = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]

        self.visited[dest.z][dest.x][dest.y] = 1

        q = deque()

        s = queueNode(dest,0)  
        q.append(s)

        while q:

            curr = q.popleft()
            point = curr.point
            # for cell in net.routed:
            #             print(cell.x,cell.y,cell.z)
            for i in range(len(net.routed)):
                if net.routed[i].x==point.x and net.routed[i].y==point.y and net.routed[i].z==point.z:
                    # print("Lets go",curr.dist)
                    return curr.dist,net.routed[i]

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

        # print(self.distMat)
        return -1,None


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
            
    def computeNet(self , dst:Cell, net:Net, idx:int):
        dist,curr_node = self.LeeAlgo(self.matr , dst, net)
        if (dist == -1):
            print(f"Path not found, skipping net {idx}: {net.cells[0].x} {net.cells[0].y} {net.cells[0].z} -> {net.cells[1].x} {net.cells[1].y} {net.cells[1].z}")
            return net

        curr_dir = 1
        while(self.distMat[curr_node.z][curr_node.x][curr_node.y] != 0):
            net.routed.append(curr_node)
            nextX , nextY , nextZ , curr_dir = self.findNextNode(self.distMat , curr_node , curr_dir)
            curr_node = Cell(nextX , nextY , nextZ)
        net.routed.append(curr_node)
        # for cells in net.routed:
        #     print(cells.x,cells.y,cells.z)
        self.distMat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        return net

    def route(self , mode = 0 , custom_order=None):
        # Mode 0 : as it is
        # Mode 1 : sort by hpwl
        # Mode 2 : sort by custom order
        nets = self.Nets
        if mode == 0:
            nets = self.Nets
        
        if mode == 1:
            nets = sorted(self.Nets, key=lambda x : x.hpwl , reverse=True)

        if mode == 2:
            nets = [self.Nets[_idx] for _idx in custom_order]






        i = 1
        n=len(nets)
        for net in nets:
            for cells in net.cells:
                self.matr[cells.z][cells.x][cells.y]=0
        for i in range(n):
            for cells in nets[i].cells:
                        self.matr[cells.z][cells.x][cells.y]=1
            for j in range(len(nets[i].cells)):
                if(j==0):
                    nets[i].routed.append(nets[i].cells[j])
                else:
                    self.computeNet(nets[i].cells[j],nets[i], i)
            for cells in nets[i].routed:
                self.matr[cells.z][cells.x][cells.y]=0
         
            
            
                 


        return nets
        
        # for net in nets:
        #     self.matr[net.src.z][net.src.x][net.src.y] = 0
        #     self.matr[net.dst.z][net.dst.x][net.dst.y] = 0

        # for net in nets:
        #     print("Routing Net " + str(i))
        #     self.matr[net.src.z][net.src.x][net.src.y] = 1
        #     self.matr[net.dst.z][net.dst.x][net.dst.y] = 1
        #     net.routed = self.computeNet(net.src , net.dst) 
        #     i+=1


    def display_curr_matr(self , nets , plot):
        mat = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
        l = 0
        # for layer in mat:
        #     print("Layer " + str(l))
        #     for line in layer:
        #         print(line)

        #     print("")
        #     l+=1
        # print("--------------------------------------------")
        vias = self.find_consecutive_coords(nets)
        print(vias)
        if plot == 2:
            j = 1
            mat2 = [[[0 for _ in range(self.dim_x)] for _ in range(self.dim_y)] for _ in range(self.dim_z) ]
            for net in nets:
                for cell in net.cells:
                    mat2[cell.z][cell.x][cell.y]=j
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
            # for layer in mat:
            #     print("Layer " + str(l))
            #     for line in layer:
            #         print(line)

            #     print("")
            #     l+=1
            # print("--------------------------------------------")

            if (plot == 1):
                for sp in range(self.dim_z):
                    plt.title("Layer " + str(sp))
                    plt.imshow(np.array(mat[sp]), cmap='viridis', interpolation='nearest')
                    plt.show()

        if (plot == 2):
            for sp in range(self.dim_z):
                plt.title("WithoutVias Layer " + str(sp))

                    # Display the image
                plt.imshow(np.array(mat[sp]), interpolation='nearest', vmin=0)
                plt.show()
            for sp in range(self.dim_z):
                plt.title("Layer " + str(sp))
                for via in vias:
                    mat[sp][via[0]][via[1]] = -1

                    # Create a custom color map
                cmap = plt.cm.viridis
                cmap.set_under('black')

                    # Display the image
                plt.imshow(np.array(mat[sp]), cmap=cmap, interpolation='nearest', vmin=0)
                plt.show()
                # plt.imshow(np.array(mat[sp]), cmap='viridis', interpolation='nearest')
                # plt.show()
    
