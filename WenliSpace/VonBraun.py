import pygame
from pygame.locals import *
import numpy as np
import random
import math
from WenliSpace import Warudo, Unit
#from Warudo import Warudo

back_color = (0,0,0)
square_color = (255, 255, 255)
obs_color = (175, 175, 175)
grid_color = (105, 105, 105)
target_color = (0, 120, 0)
    
class VonBraun():
    def __init__(self, grid_size, window_scale):
        pygame.init()
        self.grid_size = grid_size
        self.window_size = tuple(window_scale*x for x in grid_size)
        self.square_size = (math.floor(self.window_size[0]/grid_size[0]), 
                                       math.floor(self.window_size[1]/grid_size[1]))
        self.display = pygame.display.set_mode(self.window_size)
        self.warudo = Warudo.Warudo(grid_size, self.window_size)
        self.n_step = 0
        
    def set_display(self):
        pygame.display.set_caption('Electric Eye')
        self.display.fill(back_color)
        self.draw_grid()
    
    def draw_grid(self):
        n = np.prod(self.grid_size)
        for i in range(n):
            pos = (i % self.grid_size[0], math.floor(i/self.grid_size[0]))
            rect = pygame.Rect(pos[0]*self.square_size[0],pos[1]*self.square_size[1], 
                                               self.square_size[0], self.square_size[1])
            pygame.draw.rect(self.display, grid_color, rect, 1)
    
    def add_unit(self, p_x, p_y, v_range):
        self.unit = Unit.Unit(self.warudo, (p_x, p_y), v_range)
        u_rect = pygame.Rect(self.unit.x*self.square_size[0], self.unit.y*self.square_size[1],
                                                 self.unit.size[0], self.unit.size[1])
        pygame.draw.rect(self.display, square_color, u_rect)
            
    def generate_obs(self, n):
        for i in range(n):
            x, y = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
            unit = Unit.Unit(self.warudo, (x,y))
            obs_rect = pygame.Rect(unit.x*self.square_size[0], unit.y*self.square_size[1], 
                                                   self.square_size[0], self.square_size[1])
            pygame.draw.rect(self.display, obs_color, obs_rect)
            
    def set_move_target(self, mt_x, mt_y):
        mtx_old, mty_old = self.unit.set_move_target(mt_x, mt_y)
        mtx_new, mty_new = self.unit.move_target
        mt_rect_old = pygame.Rect(mtx_old*self.square_size[0], mty_old*self.square_size[1], 
                                                  self.unit.size[0], self.unit.size[1])
        mt_rect_new = pygame.Rect(mtx_new*self.square_size[0], mty_new*self.square_size[1],
                                                  self.unit.size[0], self.unit.size[1])
        pygame.draw.rect(self.display, back_color, mt_rect_old)
        pygame.draw.rect(self.display, target_color, mt_rect_new)
        
    def move_unit(self, m_x, m_y):
        ux_old, uy_old = self.unit.move(m_x, m_y)
        ux_new, uy_new = self.unit.x, self.unit.y
        u_rect_old = pygame.Rect(ux_old*self.square_size[0], uy_old*self.square_size[1], 
                                                 self.unit.size[0], self.unit.size[1])
        u_rect_new = pygame.Rect(ux_new*self.square_size[0], uy_new*self.square_size[1],
                                                 self.unit.size[0], self.unit.size[1])
        pygame.draw.rect(self.display, back_color, u_rect_old)
        pygame.draw.rect(self.display, square_color, u_rect_new)
        if (ux_old, uy_old) == (ux_new, uy_new):
            return 0
        else:
            return 1
            
    def make(self, n_obs, v_range):
        #Create obstacles
        self.n_obs = n_obs
        self.v_range = v_range
        self.generate_obs(n_obs)
        
        #Add hero unit
        px_rand, py_rand = random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1]-1)
        self.add_unit(px_rand, py_rand, v_range)
        
        #Set units target
        mtx_rand, mty_rand = random.randint(0,self.grid_size[0]-1), random.randint(0,self.grid_size[1]-1)
        while [mtx_rand, mty_rand] == [px_rand, py_rand]:
            mtx_rand, mty_rand = random.randint(0,self.grid_size[0]-1), random.randint(0,self.grid_size[1]-1)
        self.set_move_target(mtx_rand, mty_rand)
        
        #Get distance to target
        self.d_init = self.unit.get_distance()
        
        layer_1 = self.unit.get_view_grid()
        layer_2 = self.unit.get_distance_grid()
        
        return layer_1, layer_2
        
    def step(self, m_x, m_y):
        self.n_step += 1
        
        #Move unit
        self.move_unit(m_x, m_y)
        
        
        #Check if target reached
        done = False
        if [self.unit.x, self.unit.y] == self.unit.move_target:
            done = True
                        
        #Get unit sensors
        layer_1 = self.unit.get_view_grid()
        layer_2 = self.unit.get_distance_grid()
        
        #Get reward
        reward = self.get_reward()
        
        return layer_1, layer_2, done, reward
        
    def render(self):
        self.draw_grid()
        pygame.display.update()
        
    def reset(self, show_display = True):
        if show_display: self.set_display()
        self.n_step = 0
        self.warudo = Warudo.Warudo(self.grid_size, self.window_size)
        self.make(self.n_obs, self.v_range)
        layer_1 = self.unit.get_view_grid()
        layer_2 = self.unit.get_distance_grid()
        
        return layer_1, layer_2
        
    def get_reward(self):
        # ( D_init - D_step_n ) + ( N_step_min - step_n )
        return (self.d_init - self.unit.get_distance())/(self.d_init+1)