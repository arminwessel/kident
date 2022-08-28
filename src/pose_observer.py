#!/usr/bin/env python3
"""
Node to find corners of ArUcCo marker
"""
import rospy
import numpy as np
from sensor_msgs.msg import Image
from kident.msg import Obs, Meas
import cv2
import utils
from scipy.spatial.transform import Rotation
import ros_numpy
from std_msgs.msg import ByteMultiArray
import json
import pickle
import random



class PoseObserver():
    """
    Track markers and their poses
    """
    def __init__(self):
        """
        Constructor
        """
        self.observed={}
        self.tracking_queue_len=10

        self.sub_obs = rospy.Subscriber("obs", Obs, self.update_observations)
        self.pub_meas = rospy.Publisher("meas", Meas, queue_size=20)



    def update_observations(self,obs_msg):
        """
        For each marker that has been observed in the current observation, store
        the observation in the OverflowingQueue corresponding to that markers id
        If a marker does not have an associated queue yet, create a new one
        """
        obs = self.unpackage_obs_msg(obs_msg)
        id = int(obs["id"])
        if id not in self.observed:                               # previously unknown marker id 
            self.observed[id]=OverflowingQueue(self.tracking_queue_len)     # initialize a queue for it
        obs.pop("id")                                         # remove the id from the dict
        self.observed[id].enq(obs)                             # enqueue the measurement dict

    
    def process_observations(self):
        """
        Go over each marker tracked in observations and calculate the pose difference between
        the last and second to last observation in world coordinates
        A measurement contains the id of the associated marker, timestamps t- and t+ and the 
        calculated pose difference
        """
        tracked_ids=list(self.observed)
        if (tracked_ids==[]):
            return
        id = random.choice(tracked_ids)               # pick a random id
        # print("random id: {}".format(id))
        if (self.observed[id].size()<2):
            return
        obs1 = self.observed[id].deq()              # this observation is removed from queue
        obs2 = self.observed[id].peek()             # this observation will be kept
        # if ((obs2["tstamp"] - obs1["tstamp"])>0.5): # more than half a second bw frames 
        #     return
        H1=utils.H(obs1["rvec"], obs1["tvec"])
        H2=utils.H(obs1["rvec"], obs1["tvec"])
        H1i, H2i = np.linalg.inv(H1), np.linalg.inv(H2)
        dtvec = np.reshape(H1i[0:3,3]- H2i[0:3,3],(3,1))
        drvec = cv2.Rodrigues(H1i[0:3,0:3])[0] - cv2.Rodrigues(H2i[0:3,0:3])[0]


        # sanity checks
        dist_pos = np.linalg.norm(dtvec)
        if dist_pos > 0.05:
            rospy.logwarn("Distance in pose is greater than 5 cm: dist={} cm".format(100*dist_pos))
            return
        
        dist_q = np.array(obs1["joints"])-np.array(obs2["joints"])
        if np.any(dist_q > 0.01):
            rospy.logwarn("Distance q greater than 0.01 rad: dist={} rad".format(dist_q))

        m = Meas()
        m.id = id
        m.drvec = drvec
        m.dtvec = dtvec
        m.t_neg = obs1["tstamp"]
        m.t_pos = obs2["tstamp"]
        m.joints_neg = obs1["joints"]
        m.joints_pos = obs2["joints"]

        self.pub_meas.publish(m)
        
    def unpackage_obs_msg(self, msg):
        obs = {}
        obs["id"] = msg.id
        obs["rvec"] = msg.rvec
        obs["tvec"]= msg.tvec 
        obs["tstamp"] = msg.tstamp
        obs["joints"] = msg.joints
        return obs
        

class OverflowingQueue():
    """
    Custom overflowing queue. Once max number of elements is reached the oldest element is dropped 
    when adding a new element
    """
 
    # Initialize queue
    def __init__(self, max_elements):
        self.q = [None] * max_elements      # list to store queue elements
        self.capacity = max_elements        # maximum capacity of the queue
        self.front = 0                      # front points to the front element in the queue
        self.rear = -1                      # rear points to the last element in the queue
        self.count = 0                      # current size of the queue
 
    # Function to dequeue the front element
    def deq(self):
        if (self.isEmpty()):
            return None
        x = self.q[self.front]
        # print('Removing element…', x)
        self.front = (self.front + 1) % self.capacity
        self.count = self.count - 1
        # check for queue underflow
        if self.isEmpty():
            # print('Queue Underflow!! Terminating process.')
            self.front=0
            self.rear=-1
        return x
 
    # Function to add an element to the queue
    def enq(self, value):
        # check for queue overflow
        if self.isFull():
            self.deq()                      # throw out element
        # print('Inserting element…', value)
        self.rear = (self.rear + 1) % self.capacity
        self.q[self.rear] = value
        self.count = self.count + 1
 
    # Function to return the front element of the queue
    def peek(self):
        if self.isEmpty():
            # print("Can' peek empty queue")
            return None
        return self.q[self.front]
 
    # Function to return the size of the queue
    def size(self):
        return self.count
 
    # Function to check if the queue is empty or not
    def isEmpty(self):
        return self.size() == 0
 
    # Function to check if the queue is full or not
    def isFull(self):
        return self.size() == self.capacity

    # Function to return all elements in queue as ordered list
    def asList(self):
        if (self.isFull()):
            return self.q[self.front:self.capacity]+self.q[0:self.front]
        else:
            return self.q[self.front:self.rear+1]
    
    # Function to reinitialize a queue, all contents are deleted
    def drop(self):
        self.__init__(self.capacity)

# Main function.
if __name__ == "__main__":
    rospy.init_node('pose_observer')   # init ROS node named aruco_detector
    rospy.loginfo('#Node pose_observer running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass
    pe = PoseObserver()          # create instance
    
    while not rospy.is_shutdown():
        pe.process_observations()
        
