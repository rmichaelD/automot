import pygame
from pygame.locals import *
import numpy as np
import random
import math
from WenliSpace import Warudo, Unit

back_color = (0,0,0)
square_color = (255, 255, 255)
obs_color = (175, 175, 175)
grid_color = (105, 105, 105)
target_color = (0, 120, 0)

class VonBraun():
    def __init__(self, grid_size, window_scale, show_display = True):
        pygame.init()
        self.grid_size = grid_size
        self.window_size = tuple(window_scale*x for x in grid_size)
        self.square_size = (math.floor(self.window_size[0]/grid_size[0]),
                            math.floor(self.window_size[1]/grid_size[1]))
        if show_display:
            self.display = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Electric Eye')
        self.show_display = show_display
        self.warudo = Warudo.Warudo(grid_size, self.window_size)
        self.n_step = 0
        self.v_range = 4

    def draw_grid(self):
        n = np.prod(self.grid_size)
        for i in range(n):
            pos = (i % self.grid_size[0], math.floor(i/self.grid_size[0]))
            rect = pygame.Rect(pos[0]*self.square_size[0],
                               pos[1]*self.square_size[1],
                               self.square_size[0], self.square_size[1])
            pygame.draw.rect(self.display, grid_color, rect, 1)

    def add_unit(self, p_x, p_y, v_range):
        self.unit = Unit.Unit(self.warudo, (p_x, p_y), v_range)
        if self.show_display:
            u_rect = pygame.Rect(self.unit.x*self.square_size[0],
                                 self.unit.y*self.square_size[1],
                                 self.unit.size[0], self.unit.size[1])
            pygame.draw.rect(self.display, square_color, u_rect)

    def generate_obs(self, n):
        for i in range(n):
            x = random.randint(0, self.grid_size[0]-1)
            y = random.randint(0, self.grid_size[1]-1)
            if [x, y] != [self.unit.move_target[0],
                          self.unit.move_target[1]]:
                unit = Unit.Unit(self.warudo, (x, y))
                if self.show_display:
                    obs_rect = pygame.Rect(unit.x*self.square_size[0],
                                           unit.y*self.square_size[1],
                                           self.square_size[0],
                                           self.square_size[1])
                    pygame.draw.rect(self.display, obs_color, obs_rect)
            else:
                i -= 1

    def set_move_target(self, mt_x, mt_y):
        mtx_old, mty_old = self.unit.set_move_target(mt_x, mt_y)
        mtx_new, mty_new = self.unit.move_target
        if self.show_display:
            mt_rect_old = pygame.Rect(mtx_old*self.square_size[0],
                                      mty_old*self.square_size[1],
                                      self.unit.size[0], self.unit.size[1])
            mt_rect_new = pygame.Rect(mtx_new*self.square_size[0],
                                      mty_new*self.square_size[1],
                                      self.unit.size[0], self.unit.size[1])
            pygame.draw.rect(self.display, back_color, mt_rect_old)
            pygame.draw.rect(self.display, target_color, mt_rect_new)
 
    def move_unit(self, m_x, m_y):
        ux_old, uy_old = self.unit.move(m_x, m_y)
        ux_new, uy_new = self.unit.x, self.unit.y
        if self.show_display:
            u_rect_old = pygame.Rect(ux_old*self.square_size[0],
                                     uy_old*self.square_size[1],
                                     self.unit.size[0], self.unit.size[1])
            u_rect_new = pygame.Rect(ux_new*self.square_size[0],
                                     uy_new*self.square_size[1],
                                     self.unit.size[0], self.unit.size[1])
            pygame.draw.rect(self.display, back_color, u_rect_old)
            pygame.draw.rect(self.display, square_color, u_rect_new)
        if (ux_old, uy_old) == (ux_new, uy_new):
            return 0
        else:
            return 1
 
    def make(self,course = "random", n_obs = 10):
        self.n_obs = n_obs
        self.course = course
        if course == "random":
            self.make_rand_obs(n_obs)
        elif course == "classic":
            self.make_classic_obs()
        else:
            self.make_rand_obs(n_obs)

        #Get distance to target
        self.d_init = self.unit.get_distance()

        layer_1 = self.unit.get_view_grid()
        #layer_2 = self.unit.get_distance_grid()
        layer_2 = self.get_Dmt_grid()

        return layer_1, layer_2

    def make_rand_obs(self, n_obs):
        #Add hero unit
        px_rand = random.randint(0, self.grid_size[0]-1)
        py_rand = random.randint(0, self.grid_size[1]-1)
        self.add_unit(px_rand, py_rand, self.v_range)
 
        #Set units target
        mtx_rand = random.randint(0,self.grid_size[0]-1)
        mty_rand = random.randint(0,self.grid_size[1]-1)
        while [mtx_rand, mty_rand] == [px_rand, py_rand]:
            mtx_rand = random.randint(0,self.grid_size[0]-1)
            mty_rand = random.randint(0,self.grid_size[1]-1)
        self.set_move_target(mtx_rand, mty_rand)

        #Create obstacles
        self.generate_obs(n_obs)

    def make_classic_obs(self):
        #Add hero unit
        p_x, p_y = 1, self.grid_size[1] - 2
        self.add_unit(p_x, p_y, self.v_range)

        #Set units target
        mt_x, mt_y = self.grid_size[0] - 2, 1
        self.set_move_target(mt_x, mt_y)

        #Set obstacle in the x line
        for x_i in range(3, self.grid_size[0] - 4):
            y_i = 3
            unit = Unit.Unit(self.warudo, (x_i,y_i))
            obs_rect = pygame.Rect(unit.x*self.square_size[0],
                                   unit.y*self.square_size[1],
                                   self.square_size[0], self.square_size[1])
            pygame.draw.rect(self.display, obs_color, obs_rect)
 
        #Set obstacle in the y line    
        for y_i in range(3, self.grid_size[1] - 6):
            x_i = self.grid_size[0] - 4
            unit = Unit.Unit(self.warudo, (x_i,y_i))
            obs_rect = pygame.Rect(unit.x*self.square_size[0],
                                   unit.y*self.square_size[1],
                                   self.square_size[0], self.square_size[1])
            pygame.draw.rect(self.display, obs_color, obs_rect)

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
        #layer_2 = self.unit.get_distance_grid()
        layer_2 = self.get_Dmt_grid()

        #Get reward
        reward = self.get_reward()

        return layer_1, layer_2, done, reward

    def render(self):
        self.draw_grid()
        pygame.display.update()

    def reset(self):
        self.n_step = 0
        self.warudo = Warudo.Warudo(self.grid_size, self.window_size)
        if self.show_display:
            self.display.fill(back_color)
        self.make(self.course, self.n_obs)
        layer_1 = self.unit.get_view_grid()
        #layer_2 = self.unit.get_distance_grid()
        layer_2 = self.get_Dmt_grid()

        return layer_1, layer_2

    def get_reward(self):
        tolerance = 0.20
        # ( D_init - D_step_n ) + ( N_step_min - step_n )
        #dist_comp = (self.d_init - self.unit.get_distance())/(self.d_init+1)
        #return dist_comp*math.exp((-1)*self.n_step)
        d_mt = self.unit.get_distance()  # Get ditance from moving target
        dist_comp = tolerance*self.d_init + self.d_init - d_mt
        step_comp = self.n_step
        return dist_comp - step_comp

    def get_Dmt_grid(self):
        Dmt_grid_size = (self.v_range*2 + 1, self.v_range*2 + 1)
        Dmt_grid_x = np.zeros(Dmt_grid_size)
        Dmt_grid_y = np.zeros(Dmt_grid_size)
        for i in range(Dmt_grid_size[0]):
            for j in range(Dmt_grid_size[1]):
                Dmt_grid_x[i,j] = (self.unit.x + i - self.v_range)\
                                  - self.unit.move_target[0]
                Dmt_grid_y[i,j] = (self.unit.y + j - self.v_range)\
                                  - self.unit.move_target[1]
        return np.array([Dmt_grid_x, Dmt_grid_y])
