#!/usr/bin/env python3
from typing import Collection
import rospy
import numpy as np
from sensor_msgs.msg import Image
from kident.msg import Obs
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
import timeit
import scipy.signal

class SensorInput():
    """
    Detect aruco markers in image
    """

    def __init__(self,aruco_marker_length=0.12,camera_matrix=np.eye(3),camera_distortion=np.zeros(5)) -> None:
        """
        Constructor
        """
        self.pub_image_overlay = rospy.Publisher("image_overlay", Image, queue_size=20)
        self.pub_obs = rospy.Publisher("obs", Obs, queue_size=20)
        self.sub_camera_image = rospy.Subscriber("/r1/camera1/image_raw", Image, self.image_received)
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # All 5by5 Markers
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        self.aruco_length = aruco_marker_length
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        
        #rospy.wait_for_service('get_q')
        self.get_q = rospy.ServiceProxy('get_q', GetQ)
        self.q_raw = []
        self.q_filt = []
        self.freq = 5 #Hz
        sos = scipy.signal.iirfilter(10, Wn=0.01, fs=self.freq, btype="low", ftype="butter", output="sos")
        self.joint_filter = OnlineJointFilter(sos, 7)

        


    def image_received(self, image_message : Image) -> None:
        """
        Method executed for every frame: get marker observations and joint coordinates
        """
        cv_image = ros_numpy.numpify(image_message)
        joints = self.q_filt
        if (joints == None):
            rospy.logwarn("Joint coordinate readout returned None")
            return
        list_obs = self.get_observations(cv_image, rospy.get_time(), joints)
        for obs in list_obs:
            self.pub_obs.publish(self.package_obs_msg(obs))
    

    def get_observations(self, frame, tstamp, joints) -> None:
        """
        Use openCV to find ArUco markers, determine their pose and 
        """
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict)
        try: 
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 
                self.aruco_length, 
                self.camera_matrix, 
                self.camera_distortion)

            overlay_frame = cv2.aruco.drawDetectedMarkers(frame, corners,ids)
            self.pub_image_overlay.publish(ros_numpy.msgify(Image, overlay_frame.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS
            
            return  [{"id":m[0][0], "rvec":m[1].flatten().tolist(), "tvec":m[2].flatten().tolist(), "tstamp":tstamp, "joints":joints} for m in zip(ids, rvecs, tvecs)]
        except Exception as e:
            rospy.logwarn("Pose estimation failed")
            return []
    
    def processes(self):
        """
        List of scheduled processes
        """
        self.filter_joint_coordinates()

    def filter_joint_coordinates(self):
        """
        readout for joint coordinates and apply filter
        """
        # readout
        try:
            q_raw=self.get_q().q
            self.q_raw = q_raw
        except rospy.ServiceException as e:
            print("Service call to get q failed: %s"%e)

        # filtering
        q_filt = self.joint_filter(q_raw)
        self.q_filt = q_filt
    

    
    def package_obs_msg(self, obs):
        msg = Obs()
        try:
            msg.id = int(obs["id"])
            msg.rvec = obs["rvec"]
            msg.tvec = obs["tvec"]
            msg.tstamp = obs["tstamp"]
            msg.joints = obs["joints"]
        except:
            rospy.logerr("could not package observation {} of type {}".format(obs, type(obs)))
        return msg

class OnlineJointFilter():
    # stolen from https://www.samproell.io/posts/yarppg/digital-filters-python/
    def __init__(self, sos, num_joints):
        """Initialize live second-order sections filter.

        Args:
            sos (array-like): second-order sections obtained from scipy
                filter design (with output="sos").
        """
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.states = np.zeros((self.n_sections, 2, num_joints))
    
    def __call__(self, qs):
        return self.process(qs)

    def process(self, qs):
        qs_filt=[]
        if not isinstance(qs, Collection):
            qs=[qs]
        else:
            qs=list(qs)
        for i,q in enumerate(qs):
            q_filt = self.process_single(x=q, state=self.states[:,:,i])
            qs_filt.append(q_filt)
        return qs_filt

    def process_single(self, x, state):
        """Filter incoming data with cascaded second-order sections.
        """
        n_sections = self.n_sections
        sos = self.sos
        for s in range(n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = sos[s, :]

            # compute difference equations of transposed direct form II
            y = b0*x + state[s, 0]
            state[s, 0] = b1*x - a1*y + state[s, 1]
            state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.
        return y

# Main function.
if __name__ == "__main__":
    rospy.init_node('sensor_input')   # init ROS node named aruco_detector
    rospy.loginfo('#Node sensor_input running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    
    si = SensorInput()          # create instance
    r = rospy.Rate(si.freq)
    while not rospy.is_shutdown():
        starttime = rospy.get_time()
        si.processes()
        r.sleep()
        rospy.logwarn("Freq is {}".format(1/(rospy.get_time()-starttime)))
