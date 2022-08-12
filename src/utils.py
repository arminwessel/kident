#!/usr/bin/env python3
"""
Utility Functions
"""
import numpy as np
import math as m

def H(rot_euler,trans):
    rotmat=Rx(rot_euler[0]/180*np.pi)@Ry(rot_euler[1]/180*np.pi)@Rz(rot_euler[2]/180*np.pi)
    lower=np.reshape(np.array([0,0,0,1]),(1,4))
    upper=np.concatenate(
        (rotmat, np.reshape(np.array(trans),(3,1))),
        axis=1
    )
    H=np.concatenate(
        (upper, lower),
        axis=0
    )
    return np.asarray(H)


def roundprint(H):
    for line in H:
        linestr=''
        for elem in line:
            elem = 0.001*round(1000*elem)
            if (elem>=0):
                linestr += " {:.3f} ".format(elem)
            else:
                linestr += "{:.3f} ".format(elem)
        print(linestr+'\n')
    print('\n')
    
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
