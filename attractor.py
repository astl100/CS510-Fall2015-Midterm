import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import csv

class Attractor(object):
    
    def __init__(self, s=10, p=(8/3), b=28, start=0.0, end=80.0, points=10000):
        self.s = s
        self.p = p
        self.b = b
        self.params = np.array([self.s, self.p, self.b])
        
        self.start = start
        self.end = end
        self.points = points
        self.dt = np.linspace(self.start, self.end, self.points)
        self.step = (self.end - self.start) / self.points
        
        self.solution = []
        x0 = 0.0
        y0 = 0.0
        z0 = 1.0
    
    def euler(self, r):
        r = np.array([x, y, z])
        
        dx = (self.s * (y - x)) * self.dt
        dy = (x * (self.p - z) - y) * self.dt
        dz = ((x * y) - (self.b * z) * self.dt
        k1 = np.array([dx, dy, dz]) 
        
        return k1
              
        #n = len(self.dt)
        
        #for i in xrange(n - 1):
        #    dx = (self.s * (y - x)) * self.dt[i]
        #    dy = (x * (self.p - z) - y) * self.dt[i]
        #    dz = ((x * y) - (self.b * z) * self.dt[i]
        # k1 = np.array([dx, dy, dz]) 
        
    
    def rk2(self, r):
        r = np.array([x, y, z])
        new_inc = self.step/2
              
        dx = (self.s * (y - x)) * (self.step + new_inc) + (euler(r)*new_inc)
        dy = (x * (self.p - z) - y) * (self.step + new_inc) + (euler(r)*new_inc)
        dz = ((x * y) - (self.b * z) * (self.step + new_inc) + (euler(r)*new_inc)
        k2 = np.array([dx, dy, dz]) 
        
        return k2
              
        
    def rk3(self, r):
        r = np.array([x, y, z])
        new_inc = self.step/2
              
        dx = (self.s * (y - x)) * (self.step + new_inc) + (rk2(r)*new_inc)
        dy = (x * (self.p - z) - y) * (self.step + new_inc) + (rk2(r)*new_inc)
        dz = ((x * y) - (self.b * z) * (self.step + new_inc) + (rk2(r)*new_inc)
        k3 = np.array([dx, dy, dz]) 
        
        return k3
              
    def rk4():
        r = np.array([x, y, z])
        new_inc = self.step/2
              
        dx = (self.s * (y - x)) * (self.step + new_inc) + (rk3(r)*self.step)
        dy = (x * (self.p - z) - y) * (self.step + new_inc) + (rk3(r)*self.step)
        dz = ((x * y) - (self.b * z) * (self.step + new_inc) + (rk3(r)*self.step)
        k4 = np.array([dx, dy, dz]) 
        
        return k4
              
        
    def evolve(self, r0 = np.array([x0,y0,z0]), order = 4):
        t = 0
        if order == 1:
              i = euler(r0)
        elif order == 2:
              i = rk2(r0)
        else:
              i = rk4(r0)
        #t = [t+i for _ in self.dt]
        df = pd.DataFrame({[self.dt, i]})
        df.columns = ["t", "x", "y", "z"]
        self.solution = df
        return self.solution
              
    
    def save(self):
        self.solution.to_csv('solution.csv')
        
    def plotx(self):
        plt.plot(self.dt, dx)
        
    def ploty(self): 
        plt.plot(self.dt, dy)
    
    def plotz(self):
        plt.plot(self.dt, dz) 
    
    def plot3d(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.plot(dx, dy, dz)#each being a list of values calculated from appropriate dt values
        plt.show()