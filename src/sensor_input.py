#!/usr/bin/env python3
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
        


    def image_received(self, image_message : Image) -> None:
        """
        Method executed for every frame: get marker observations and joint coordinates
        """
        cv_image = ros_numpy.numpify(image_message)
        joints = self.get_joint_coordinates()
        list_obs = self.get_observations(cv_image, rospy.get_time(), joints)
        for obs in list_obs:
            self.pub_obs.publish(self.package_obs_msg(obs))
    

    def get_observations(self, frame, tstamp, joints) -> None:
        """
        Use openCV to find ArUco markers, determine their pose and 
        """
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict)
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 
            self.aruco_length, 
            self.camera_matrix, 
            self.camera_distortion)

        overlay_frame = cv2.aruco.drawDetectedMarkers(frame, corners,ids)
        self.pub_image_overlay.publish(ros_numpy.msgify(Image, overlay_frame.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS
        return  [{"id":m[0][0], "rvec":m[1].flatten().tolist(), "tvec":m[2].flatten().tolist(), "tstamp":tstamp, "joints":joints} for m in zip(ids, rvecs, tvecs)]
    
    def get_joint_coordinates(self):
        """
        TODO implement readout for joint coordinates
        """
        q = [1,2,3,4,5,6,7]       # Dummy
        return q
    
    def package_obs_msg(self, obs):
        msg = Obs()
        msg.id = int(obs["id"])
        msg.rvec = obs["rvec"]
        msg.tvec = obs["tvec"]
        msg.tstamp = obs["tstamp"]
        msg.joints = obs["joints"]
        return msg
        

# Main function.
if __name__ == "__main__":
    rospy.init_node('sensor_input')   # init ROS node named aruco_detector
    rospy.loginfo('#Node sensor_input running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    si = SensorInput()          # create instance
    
    rospy.spin()
