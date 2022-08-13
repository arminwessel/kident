#!/usr/bin/env python3

import time
import arcpy
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from kident.srv import GetQ, GetQResponse


class MoveIiwa():

    def __init__(self) -> None:
        self.netif_addr = '127.0.0.1'
        # netif_addr = '192.168.1.3'
        self.iiwa = arcpy.Robots.Iiwa(self.netif_addr)
        time.sleep(0.1)
        self.q = self.iiwa.model.get_q()
        self.pub_q=rospy.Publisher("iiwa_q", Float32MultiArray, queue_size=20)
        self.serv_q = rospy.Service('get_q', GetQ, self.get_q)

    def move_random(self, T_traj = 8):  
        t0 = self.iiwa.get_time()
        q1 = self.q + 0.5*np.random.rand(7,1)
        self.iiwa.move_jointspace(q1, t0, T_traj)
        time.sleep(T_traj)
        self.q = self.get_q()

    def get_q(self):
        res = GetQResponse()
        res.q = self.iiwa.model.get_q()
        return res


# Main function.
if __name__ == "__main__":
    rospy.init_node('sensor_input')   # init ROS node named aruco_detector
    rospy.loginfo('#Node sensor_input running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    mi = MoveIiwa()          # create instance
    
    while not rospy.is_shutdown():
        mi.move_random()
