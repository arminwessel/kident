#!/usr/bin/env python3
"""
Estimate the kinematic model error of a robot manipulator
The model is based on the DH convention
""" 
# import statements
import json
import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from std_msgs.msg import String
#from geometry_msgs.msg import Twist, Pose
from kident.msg import Est, Meas
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

        self.sub_meas = rospy.Subscriber("meas", Meas, self.process_measurement)
        self.pub_est = rospy.Publisher("est", Est, queue_size=20)


        self.rls=RLS(28,1) # estimate 28 params, forgetting factor none

        # nominal DH parameters
          # nominal theta parameters are joint coors
        d_nom=np.array([0,0,0.42,0,0.4,0,0])
        a_nom=np.array([0,0,0,0,0,0,0])
        alpha_nom=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2])


    
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
    

    def process_measurement(self, m):
        print("test")

        # calculate the expected pose difference based on forward kinematics with nominal params
        theta_nom1 = np.array(m.joints_neg).flatten()
        theta_nom2 = np.array(m.joints_pos).flatten()
        T_nom1 = self.get_T__i0(7,theta_nom1, self.d_nom, self.a_nom, self.alpha_nom)
        T_nom2 = self.get_T__i0(7,theta_nom2, self.d_nom, self.a_nom, self.alpha_nom)
        dtvec_nom = T_nom1[0:3,3].reshape((3,1)) - T_nom2[0:3,3].reshape((3,1))
        drvec_nom = cv2.Rodrigues(T_nom1[0:3,0:3]) - cv2.Rodrigues(T_nom2[0:3,0:3])

        # extract the measured pose differences
        dtvec_real = m.dtvec
        drvec_real = m.drvec

        # calculate parameter jacobian for 0.5*((q+) + (q-))
        jacobian = self.get_parameter_jacobian(0.5*(theta_nom1+theta_nom2), self.d_nom, self.a_nom, self.alpha_nom)

        # calculate errors between expected and measured pose difference
        dtvec_error = np.reshape(dtvec_real-dtvec_nom,(3,1))
        drvec_error = np.reshape(drvec_real-drvec_nom,(3,1))
        current_error=np.concatenate((dtvec_error, drvec_error),axis=0)

  
        # use RLS to improve estimate of parameters
        self.rls.add_obs(S=jacobian, Y=current_error)
        estimate_k = self.rls.get_estimate().flatten()

        msg = Est()
        msg.estimate = estimate_k
        msg.joints = theta_nom1

        self.pub_est(msg)


class RLS(): 
    def __init__(self, num_params, q, alpha=1e3)->None:
        """
        num_params: number of parameters to be estimated
        q: forgetting factor, usually very close to 1.
        alpha: initial value on idagonal of P
        """
        assert q <= 1 and q > 0.9, "q usually needs to be from ]0.9, 1]"
        self.q = q

        self.num_params = num_params
        self.P = alpha*np.eye(num_params) #initial value of matrix P
        self.phat = np.zeros((num_params,1)) #initial guess for parameters, col vector
        self.num_obs=0
        

    def add_obs(self, S, Y)->None:
        """
        Add an observation
        S_T: array of data vectors [[s1],[s2],[s3]...]
        Y: measured outputs vector [[y1],[y2],[y3]...]
        """
        if S.ndim==1: # 1D arrays are converted to a row in a 2D array
            S = np.reshape(S,(1,-1))
        if Y.ndim==1:
            Y = np.reshape(Y,(-1,1))

        assert np.shape(S)[1]==self.num_params, "number of parameters has to agree with measurement dim"
        assert np.shape(S)[0]==np.shape(Y)[0], "observation dimensions don't match"



        for obs in zip(S,Y): # iterate over rows, each iteration is an independent measurement
            (s_T, y)=obs
            s_T = np.reshape(s_T,(1,-1))
            s=np.transpose(s_T)
            _num=self.P@s
            _den=(self.q + s_T@self.P@s)
            self.k = _num/_den

            self.P = (self.P - self.k@s_T@self.P)*(1/self.q)

            self.phat = self.phat + self.k*(y - s_T@self.phat)
            self.num_obs = self.num_obs+1

    def get_estimate(self):
        return self.phat

    def get_num_obs(self):
        return self.num_obs



class Kalman: 
    def __init__(self):
        '''
        TODO
        '''
        pass

    def add_obs(self):
        '''
        TODO
        '''            
        pass
    def get_estimate(self):
        '''
        TODO
        '''
        pass


    def get_num_obs(self):
        '''
        TODO
        '''
        pass


# Main function.
if __name__ == "__main__":
    rospy.init_node('dh_estimator')   # init ROS node
    rospy.loginfo('#Node dh_estimator running#')
    while not rospy.get_rostime():      # wait for ros time service
        pass
    dhe = DHestimator()          # create instance
    print("instance created")
    rospy.spin()
