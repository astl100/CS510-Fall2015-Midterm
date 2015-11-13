#clarify the evolve function so that you can calculate values for the test file
    #fix tests and calc values to compare to
#do ExploreAttractor and make pretty graphs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import csv

class Attractor(object):
    
    def __init__(self, s=10, b=(8/3), p=28, start=0.0, end=80.0, points=10000):
        """Initializes all needed variables"""
        self.s = s
        self.p = p
        self.b = b
        self.params = np.array([self.s, self.p, self.b])
        
        self.start = start
        self.end = end
        self.points = points
        
        #make sure dt is a value, not a list
        self.dt = (self.end - self.start) / self.points
        
        self.solution = []
        
        #generate arrays to commit values to and later plot
        self.t_list = np.linspace(self.start, self.end, self.points)
        self.x = np.zeros(self.points)
        self.y = np.zeros(self.points)
        self.z = np.zeros(self.points)
    
    def euler(self, r, dt=0.0):
        """Calculates the change in x,y,z based on the initialized s,b,p values and calculated dt value
        Uses the Euler method to generate"""
        x, y, z = r
        if not dt: dt = self.dt
        
        dx = (self.s * (y - x)) * dt
        dy = (x * (self.p - z) - y) * dt
        dz = ((x * y) - (self.b * z)) * dt 
        
        k1 = np.array([dx, dy, dz])
        
        return k1

    
    def rk2(self, r):
        """Calculates the second order runge-kutta increment from initialized values(x, y, z, s, b, p, dt)"""
        x, y, z = r

        #Calculating k1
        k1 = self.euler(r, dt = self.dt/2.0)
        r_prime = r + k1
        #Calculating k2
        k2 = self.euler(r_prime, dt = self.dt)

        return k2


    def rk4(self, r):
        """Calculates the fourth order runge-kutta increment from initialized values(x, y, z, s, b, p, dt)"""
        x, y, z = r

        #Calculating k1
        k1 = self.euler(r, dt = self.dt)
        r_prime = r + k1/2.0
        #Calculating k2
        k2 = self.euler(r_prime, dt = self.dt/2.0)
        r2_prime = r + k2/2.0
        k3 = self.euler(r2_prime, dt = self.dt/2.0)
        r3_prime = r +k3
        k4 = self.euler(r3_prime, dt = self.dt)

        return k4


    def evolve(self, r0 = [0.1, 0.0, 0.0], order = 4):
        """Generates lists of calculates x, y, and z based on each changing step
        Calculated values depend on order/method chosen"""
        x0, y0, z0 = r0
        #if not dt: dt = self.dt

        if order == 1:
              inc = self.euler
        elif order == 2:
              inc = self.rk2
        elif order == 4:
              inc = self.rk4
        else:
              print "Incorrect input"
        #t = [t+i for _ in self.dt]

        self.x[0] = x0
        self.y[0] = y0
        self.z[0] = z0
        
        for i in xrange(0, self.t_list.size-1):
            dx, dy, dz = inc(np.array([self.x[i], self.y[i], self.z[i]]))
            self.x[i+1] = self.x[i] + dx
            self.y[i+1] = self.y[i] + dy
            self.z[i+1] = self.z[i] + dz
        df = pd.DataFrame({'t': self.t_list, 'x': self.x, 'y': self.y, 'z': self.z})
        self.solution = df

        return self.solution

    
    def save(self):
        """Saves datafram of solutions to 'solutions.csv'"""
        self.solution.to_csv('solution.csv')
        
    def plotx(self):
        """Plots x(t) curve of the solution VS time"""
        plt.plot(self.t_list, self.solution['x'])
        
    def ploty(self): 
        """Plots y(t) curve of the solution VS time"""
        plt.plot(self.t_list, self.solution['y'])
    
    def plotz(self):
        """Plots z(t) curve of the solution VS time"""
        plt.plot(self.t_list, self.solution['z']) 
              
    def plotxy(self):
        """Plots x-y planar projections of the solution curve"""
        plt.plot(self.solution['x'], self.solution['y'])
        
    def plotyz(self):
        """Plots y-z planar projections of the solution curve"""
        plt.plot(self.solution['y'], self.solution['z'])
              
    def plotzx(self):
        """Plots z-x planar projections of the solution curve"""
        plt.plot(self.solution['z'], self.solution['x'])
    
    def plot3d(self):
        """Generates full 3d plot of x-y-z solution curves"""
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        ax.plot(self.solution['x'], self.solution['y'], self.solution['z'])
        #each being a list of values calculated from appropriate dt values
        plt.show()