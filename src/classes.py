import numpy as np
import random
import math

class Macro:
    def __init__(self, name , id, w , h , pins , priority=0):
        self.name = name
        self.id = id
        self.w = w
        self.h = h
        self.pins = pins
        self.priority = priority
        self.x = 0
        self.y = 0
        self.orientation = 1 # 1,2,-1,-2

class Net:
    def __init__(self , name , macros, priority=0):
        self.name = name
        self.priority = priority
        self.macros = macros
        self.routed_cells = []
        self.pins = [] # [(pin1 , macro1) , (pin2 , macro2) ,..]

class Floor:
    def __init__(self , w , h , gridUnit , layers = 4):
        self.w = w
        self.h = h
        self.gridUnit = gridUnit
        self.layers = layers
        
class Pin:
    def __init__(self,x,y):# x,y are relative positions
        self.x = x
        self.y = y



