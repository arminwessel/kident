#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from kident.msg import Obs, Meas
# from geometry_msgs.msg import Point, Quaternion, Pose
# from kident.msg import Point2DArray
import cv2
# import utils
# from scipy.spatial.transform import Rotation
import ros_numpy
import json
import pickle
import arcpy
from kident.srv import GetQ
import math
import time


class InputSimulator():
    """
    Generate data pairs of joints state and pose error 
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        self.pub_meas = rospy.Publisher("meas", Meas, queue_size=20)

        self.d_nom=np.array([0,0,0.42,0,0.4,0,0.08]) # d6 is 8cm because of camera
        self.a_nom=np.array([0,0,0,0,0,0,0])
        self.alpha_nom=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2])

        self.d_real=self.d_nom + np.array([0,0,0,0.07,0,0,0.0]) # error on d3
        self.a_real=self.a_nom + np.array([0,0,0,0,0,0,0.0])
        self.alpha_real=self.alpha_nom + np.array([0,0,0,0,0,0,0])


    
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


    def simulate_measurement(self) -> None:
        """
        simulates a measurement
        """
        m = Meas()
        m.t_neg = rospy.get_time()
        m.joints_neg = self.get_joint_coordinates()
        T_real = self.get_T__i0(7, np.array(m.joints_neg), self.d_real, self.a_real, self.alpha_real)
        real_pos1 = T_real[0:3,3].reshape((3,1))
        real_rot1 = T_real[0:3,0:3]

        time.sleep(0.001)

        m.t_pos = rospy.get_time()
        m.joints_pos = self.get_joint_coordinates()
        T_real = self.get_T__i0(7, np.array(m.joints_pos), self.d_real, self.a_real, self.alpha_real)
        real_pos2 = T_real[0:3,3].reshape((3,1))
        real_rot2 = T_real[0:3,0:3]
        
        m.dtvec = real_pos2 - real_pos1
        m.drvec = cv2.Rodrigues(real_rot2)[0] - cv2.Rodrigues(real_rot1)[0]
        m.id = 777

        self.pub_meas.publish(m)
    
    def get_joint_coordinates(self):
        """
        readout for joint coordinates
        """
        try:
            rospy.wait_for_service('get_q')
            get_q = rospy.ServiceProxy('get_q', GetQ)
            res=get_q()
            return res.q
        except rospy.ServiceException as e:
            print("Service call to get q failed: %s"%e)

    

# Main function.
if __name__ == "__main__":
    rospy.init_node('input_simulator')   # init ROS node named aruco_detector
    rospy.loginfo('#Node input simulator running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    ins = InputSimulator()          # create instance
    
    while not rospy.is_shutdown():
        ins.simulate_measurement()
        time.sleep(0.1)
