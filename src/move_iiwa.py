#!/usr/bin/env python3

import time
import arcpy
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from kident.srv import GetQ, GetQResponse
import socket

class MoveIiwa():

    def __init__(self) -> None:
        self.netif_addr = '127.0.0.1' # loopback to gazebo
        # netif_addr = '192.168.1.3'
        try:
            self.iiwa = arcpy.Robots.Iiwa(self.netif_addr)
        except Exception as e:
            rospy.logerr("Instance of Iiwa robot could not be created: {}".format(e))
            pass
        time.sleep(0.1)
        self.q = self.iiwa.model.get_q()
        self.pub_q=rospy.Publisher("iiwa_q", Float32MultiArray, queue_size=20)
        self.serv_q = rospy.Service('get_q', GetQ, self.get_q_serv)

    def move_random(self, T_traj = 8):  
        t0 = self.iiwa.get_time()
        q1 = self.q + 0.5*np.random.rand(7,1)
        self.iiwa.move_jointspace(q1, t0, T_traj)
        time.sleep(T_traj)
        self.q = self.iiwa.model.get_q()

    def move_normal(self, T_traj = 7):  
        t0 = self.iiwa.get_time()
        q1 = (np.array(self.q) + np.random.default_rng().normal(0, 0.1, (7,1)))
        self.iiwa.move_jointspace(q1, t0, T_traj)
        time.sleep(T_traj)
        self.q = self.iiwa.model.get_q()
    
    
    def move_linear(self, T_traj = 5):
        t0 = self.iiwa.get_time()
        q1 = self.q + np.pi/6*np.ones((7,1)) - 0.5*np.random.rand(7,1)
        self.iiwa.move_jointspace(q1, t0, T_traj)
        time.sleep(T_traj)
        self.q = self.iiwa.model.get_q()
        t0 = self.iiwa.get_time()
        q1 = self.q - np.pi/6*np.ones((7,1)) + 0.5*np.random.rand(7,1)
        self.iiwa.move_jointspace(q1, t0, T_traj)
        time.sleep(T_traj)
        self.q = self.iiwa.model.get_q()
        pass

    def get_q_serv(self,req):
        res = GetQResponse()
        res.q = self.iiwa.model.get_q()
        return res


# Main function.
if __name__ == "__main__":
    rospy.init_node('move_iiwa')   # init ROS node named aruco_detector
    rospy.loginfo('#Node move iiwa running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    mi = MoveIiwa()          # create instance
    
    while not rospy.is_shutdown():
        mi.move_normal()
