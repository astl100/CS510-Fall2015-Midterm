from attractor import Attractor
import numpy as np
import pandas as pd
import csv

def test_dt():
    """Tests step Value"""
    a = Attractor()
    dt_true = (80.0-0.0)/10000

    print ("Actual dt value: ", a.dt)
    print ("Expected dt value: ", dt_true)
    assert a.dt == dt_true


def test_euler():
    """Tests if dx from euler method is implemented properly
    Uses set x, y, and z to be 0.1, 0.0, 0.0"""
    a = Attractor()
    #say x, y, z = [0.1, 0.0, 0.0]

    dx = (10 * (0.0 - 0.1)) * (80.0-0.0)/10000
    dy = (0.1 * (28 - 0.0) - 0.0) * (80.0-0.0)/10000
    dz = ((0.1 * 0.0) - (8/3 * 0.0)) * (80.0-0.0)/10000
    ex_euler = np.array([dx, dy, dz])

    print ("Actual increments: ", a.euler([0.1, 0.0, 0.0]))
    print ("Expected increments: ", ex_euler)
    assert a.euler([0.1, 0.0, 0.0])[0] == ex_euler[0]

def test_x_generate():
    """Tests if evolve method is implemented properly
    Uses set x, y, and z to be 0.1, 0.0, 0.0"""
    a = Attractor()
    #say x, y, z = [0.1, 0.0, 0.0]

    dx = (10 * (0.0 - 0.1)) * (80.0-0.0)/10000 + 0.1
    dy = (0.1 * (28 - 0.0) - 0.0) * (80.0-0.0)/10000 + 0.0
    dz = ((0.1 * 0.0) - (8/3 * 0.0)) * (80.0-0.0)/10000 + 0.0
    ex_1 = np.array([dx, dy, dz])

    dx2 = (10 * (dy - dx)) * (80.0-0.0)/10000 + dx 
    dy2 = (dx * (28.0 - dz) - dy) * (80.0-0.0)/10000 + dy
    dz2 = ((dx * dy) - (8/3 * dz)) * (80.0-0.0)/10000 + dz
    ex_2 = np.array([dx2, dy2, dz2])

    dx3 = (10 * (dy2 - dx2)) * (80.0-0.0)/10000 + dx2
    dy3 = (dx2 * (28.0 - dz2) - dy2) * (80.0-0.0)/10000 + dy2
    dz3 = ((dx2 * dy2) - (8/3 * dz2)) * (80.0-0.0)/10000 + dz2
    ex_3 = np.array([dx3, dy3, dz3])

    dx4 = (10 * (dy3 - dx3)) * (80.0-0.0)/10000 + dx3
    dy4 = (dx3 * (28 - dz3) - dy3) * (80.0-0.0)/10000 + dy3
    dz4 = ((dx3 * dy3) - (8/3 * dz3)) * (80.0-0.0)/10000 + dz3
    ex_4 = np.array([dx4, dy4, dz4])

    dx5 = (10 * (dy4 - dx4)) * (80.0-0.0)/10000 + dx4
    dy5 = (dx4 * (28 - dz4) - dy4) * (80.0-0.0)/10000 + dy4
    dz5 = ((dx4 * dy4) - (8/3 * dz4)) * (80.0-0.0)/10000 + dz4
    ex_5 = np.array([dx5, dy5, dz5])

    print a.solution['x']
    
    #print ("Actual increments: ", str(a.solution()['x']).strip('[]'))
    #print ("Expected increments: ", ex_5)
    #assert a.solution()['x'] == ex_5[0]

    #print "Expected increments: " + str([dx, dx2, dx3, dx4, dx5]).strip('[]')
    #print ("Actual increments: ", x[:5])
    #assert x[:5] == str([dx, dx2, dx3, dx4, dx5]).strip('[]')