import numpy as np
import math

class Warudo:
    def __init__(self, grid_size, window_size):
        self.grid_size = grid_size
        self.square_size = (math.floor(window_size[0]/grid_size[0]), math.floor(window_size[1]/grid_size[1]))
        self.units = np.zeros(grid_size, dtype = int)
        #self.ids = np.zeros(grid_size, dtype = int)
    
    def get_units(self):
        return tuple(self.units)
        
    def add_unit(self, unit):
        self.units[unit.x, unit.y] = 1
        #self.ids[unit.x, unit.y] = unit.id
        
    def set_display(self, display):
        self.display = display
        
    def move_unit(self, x_old, y_old, unit):
        self.units[x_old, y_old] = 0
        #self.ids[x_old, y_old] = 0
        self.add_unit(unit)