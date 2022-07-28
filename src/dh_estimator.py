#!/usr/bin/env python3
"""
Estimate the kinematic model error of a robot manipulator
The model is based on the DH convention
""" 
# import statements
from re import I
import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
#from geometry_msgs.msg import Twist, Pose
#from vservo.msg import  Point2DArray
#from std_msgs.msg import Float32
#from std_srvs.srv import SetBool, SetBoolResponse
#import utils

class DHestimator():
    """
    Estimate the kinematic model error of a robot manipulator
    The model is based on the DH convention
    """

    def __init__(self) -> None:
        """
        Constructor
        """
    
    def get_T__i(self, theta__i, d__i, a__i, alpha__i) -> np.array:
        t1 = math.cos(theta__i)
        t2 = math.sin(theta__i)
        t3 = math.cos(alpha__i)
        t4 = math.sin(alpha__i)
        return np.array([[t1,-t2 * t3,t2 * t4,t1 * a__i],[t2,t1 * t3,-t1 * t4,t2 * a__i],[0,t4,t3,d__i],[0,0,0,1]])


    def get_T_jk(self,j,k,theta_all, d_all, a_all, alpha_all) -> np.array:
        """
        T_jk = T^j_k
        """
        theta_all, d_all, a_all, alpha_all = theta_all.flatten(), d_all.flatten(), a_all.flatten(), alpha_all.flatten()
        assert theta_all.size==7 and d_all.size==7 and a_all.size==7 and alpha_all.size==7, "DH param vector len"
        T=np.eye(4)
        for i in range(k+1, j+1, 1): # first i=k+1, last i=j
            T=np.matmul(T,self.get_T__i(theta_all[i-1], d_all[i-1], a_all[i-1], alpha_all[i-1]))
        return T

    def get_T__i0(self, i, theta_all, d_all, a_all, alpha_all) -> np.array:
        return self.get_T_jk(i,0,theta_all, d_all, a_all, alpha_all)
    
    
    def get_parameter_jacobian(self, theta_all, d_all, a_all, alpha_all) -> np.array:
        W1 = W2 = W3 = W4 = W7 = W8 = np.zeros((3,0))
        T__7_0=self.get_T__i0(7, theta_all, d_all, a_all, alpha_all)
        t__7_0=T__7_0[0:3,3]
        for i in range(1,8): # i=1..7
            T__i_0=self.get_T__i0(i-1, theta_all, d_all, a_all, alpha_all)
            t__i_0=T__i_0[0:3,3]
            R__i_0=T__i_0[0:3,0:3]
            m__1i=np.array([[0],[0],[1]])
            m__2i=np.array([[math.cos(theta_all[i-1])],[math.sin(theta_all[i-1])],[0]])
            m__3i=np.array([[-d_all[i-1]*math.sin(theta_all[i-1])],[d_all[i-1]*math.cos(theta_all[i-1])],[0]])
            _t1=np.matmul(R__i_0,m__1i)
            _t2=np.matmul(R__i_0,m__2i)

            _w=np.reshape(np.cross(t__i_0,_t1.flatten()),(3,1))
            W1 = np.concatenate((W1,_w), axis=1)
            
            W2 = np.concatenate((W2,_t1), axis=1)

            W3 = np.concatenate((W3,_t2), axis=1)

            _w=np.reshape(np.cross(t__i_0,_t2.flatten()),(3,1))+np.matmul(R__i_0,m__3i)
            W4 = np.concatenate((W4,_w),axis=1)

            _w = np.reshape(np.cross(_t1.flatten(),t__7_0),(3,1))+np.reshape(W1[:,-1],(3,1))
            W7 = np.concatenate((W7,_w),axis=1)

            _w=np.reshape(np.cross(_t2.flatten(),t__7_0),(3,1))+np.reshape(W4[:,-1],(3,1))
            W8=np.concatenate((W8,_w),axis=1)
        J = np.zeros((6,28))
        J[0:3,:]=np.concatenate((W7, W2, W3, W8), axis=1)
        J[3:6,:]=np.concatenate((W2, np.zeros((3,7)), np.zeros((3,7)), W3), axis=1)
        return J

    

