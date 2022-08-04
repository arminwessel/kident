#!/usr/bin/env python3
"""
Node to find corners of ArUcCo marker
"""
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Quaternion, Pose
from kident.msg import Point2DArray
import cv2
import utils
from scipy.spatial.transform import Rotation
import ros_numpy


class PoseEstimator():
    """
    Track markers and theit poses, calculate camera pose
    """
    def __init__(self):
        """
        Constructor
        """
        self.pose={}
        self.pose['r'] = np.zeros(3)
        self.pose['t'] = np.zeros(3)

        self.tracked={}

        self.queue_len=5
        # initialize pose, all measurements are relative pose differences
        # initialize dictionary of PoseQueues to store measurement data
        pass

    def add_measurements(self,measurements):
        """
        Add a number of measurements: ids and poses and timestamp of image frame
        """
        for m in measurements:
            id = m["id"]
            if id not in self.tracked:                          # previously unknown marker id 
                self.tracked[id]=PoseQueue(self.queue_len)      # initialize a queue for it
            self.tracked[id].enq(m.pop("id"))                     # enqueue the measurement dict without its own id key
        pass

    def estimate_cam_pose():
        """
        Based on the pose differences between markers of current, previous and even older frames,
        calculate most likely movement of the camera between each pair of frames. Aggregating these pose
        changes calculate the most likely current camera pose.

        In essence use a number of increasingly old measurements to get a position of the camera. 
        Store in a long PoseQueue
        """


class PoseQueue():
    """
    Custom overflowing queue. Once max number of elements is reached the oldest element is dropped
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
            print("Can' peek empty queue")
            exit(-1)
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
    pe = PoseEstimator()  # create instance
    ms=[]
    ms.append({"id":0, "rvec":'r1', "tvec":'r1', "tstamp":"TODO timestamp"})
    ms.append({"id":1, "rvec":'r2', "tvec":'r2', "tstamp":"TODO timestamp"})
    pe.add_measurements(ms)
    for id in pe.tracked:
        print(pe.tracked[id].asList())
