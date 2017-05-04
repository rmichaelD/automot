import numpy as np

#unique_ids = random.sample(range(1, 10000), 1000)

class Unit:
    def __init__(self, Warudo, spawn_coord, v_range = 0, move_target = (-1,-1)):
        self.warudo = Warudo
        #self.id = unique_ids.pop()
        self.size = (self.warudo.square_size[0], self.warudo.square_size[1])
        self.x, self.y = list(spawn_coord)
        self.warudo.add_unit(self)
        self.v_range = v_range
        if move_target == (-1,-1):
            self.move_target = [self.x, self.y]
        else:
            self.move_target = list(move_target)
            
    def get_view_grid(self):
        V_grid_size = (self.v_range*2 + 1, self.v_range*2 + 1)
        V_grid = np.zeros(V_grid_size, dtype = float)
        for i in range(V_grid_size[0]):
            for j in range(V_grid_size[1]):
                V_grid_x = self.x + i - self.v_range
                V_grid_y = self.y + j - self.v_range
                if V_grid_x < 0 or V_grid_x >= self.warudo.grid_size[0] or V_grid_y < 0 or V_grid_y >= self.warudo.grid_size[1]:
                    V_grid[i,j] = 1
                else:
                    V_grid[i,j] = self.warudo.units[V_grid_x, V_grid_y]
        return V_grid
    
    def get_view_grid_cirlce(self):
        v_range = self.v_range
        v_point = [(0,0)]
        points = []
        points.extend(v_point)           
        for n in range(v_range):
            next_iter = []
            for point in points:
                x, y = point
                new_point = []
                for (d_x, d_y) in [(0,1), (1,0), (0,-1), (-1,0)]:
                    new_point.append((x+d_x, y+d_y))
                new_point = set(new_point) - set(v_point)
                next_iter.extend(new_point)
                v_point.extend(new_point)
            points = next_iter
        
        features = []
        for point in v_point:
            rel_point = (self.x+point[0], self.y+point[1])
            if rel_point[0] < 0 or rel_point[0] >= self.warudo.grid_size[0] or rel_point[1] < 0 or rel_point[1] >= self.warudo.grid_size[1]:
                obj = -1
            else:
                obj = self.warudo.units[rel_point[0], rel_point[1]]
            features.append((point[0], point[1], obj))
        #Return a list of vectors
        # Vector := [x, y, object (0 or 1)]
        obj_grid = np.ones([v_range*2 + 1, v_range*2 + 1])*-1
        for (i, j, obj) in features:
            obj_grid[i+v_range,j+v_range] = obj
        
        return obj_grid
        
    def get_distance(self, x = -1, y = -1):
        if (x,y) == (-1, -1):
            x, y = self.x, self.y
        D_max = max(abs(x - self.move_target[0]), abs(y - self.move_target[1]))
        return D_max
        
    def get_distance_grid(self):
        D_grid_size = (self.v_range*2 + 1, self.v_range*2 + 1)
        D_grid = np.zeros(D_grid_size, dtype = float)
        for i in range(D_grid_size[0]):
            for j in range(D_grid_size[1]):
                D_grid_x = self.x + i - self.v_range
                D_grid_y = self.y + j - self.v_range
                if D_grid_x < 0 or D_grid_x >= self.warudo.grid_size[0] or D_grid_y < 0 or D_grid_y >= self.warudo.grid_size[1]:
                    D_grid[i,j] = self.get_distance(D_grid_x, D_grid_y)
                else:
                    D_grid[i,j] = self.get_distance(D_grid_x, D_grid_y)
        return D_grid
        
    def valide_move_target(self, tx_new, ty_new):
        if tx_new < 0 or ty_new < 0 or tx_new >= self.warudo.grid_size[0] or ty_new >= self.warudo.grid_size[1]:
            return self.move_target
        elif self.warudo.units[tx_new, ty_new] == 1:
            return self.move_target
        else:
            return [tx_new, ty_new]
            
    def set_move_target(self, t_x, t_y):
        tx_old, ty_old = t_x, t_y
        self.move_target = self.valide_move_target(t_x, t_y)
        return tx_old, ty_old
            
    def valide_move(self, x_new, y_new):
        if x_new < 0 or y_new < 0 or x_new >= self.warudo.grid_size[0] or y_new >= self.warudo.grid_size[1]:
            return self.x, self.y
        elif self.warudo.units[x_new, y_new] == 1:
            return self.x, self.y
        else:
            return x_new, y_new
        
    def move(self, m_x, m_y):
        x_old, y_old = self.x, self.y
        self.x, self.y = self.valide_move(m_x + self.x, m_y + self.y)
        return x_old, y_old
        
