#!/usr/bin/env python3
"""
Utility Functions
"""

from vservo.msg import Point2D, Point2DArray
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
import rospy
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation
import cv2

def conv_Point2DArray_2_nparray(data : Point2DArray) -> np.array:
    if not hasattr(data,'points'):
        return None
    _points = data.points

    if len(_points)==0:
        return None

    out = np.zeros([0,2])               # initialize empty matrix with 2 cols
    for point in _points:
        out=np.append(out,[[point.x, point.y]], axis=0)
    out = np.array(out,dtype=np.int32)
    return out

def conv_nparray_2_Point2DArray(data: np.array) -> Point2DArray:
    out = Point2DArray()
    idx = None
    for idx, row in enumerate(data): # idx is indexing the elements in the loop starting with 0
        [_x,_y] = row
        point = Point2D()
        point.x ,point.y =_x, _y
        out.points.append(point)
    if idx == None:
        out.length=None
    else:
        out.length=idx+1 # final value of idx is last index, add one to get number of iterations on for loop
    return out

def tf_cc(coordinates : np.array) -> np.array:
    """
    Transform Camera Center Coordinates
    """
    if not np.shape(coordinates) == (4,2):
        return np.zeros((0,2))
    width, height = 640, 480 # x=0..640, y=0..480
    return np.subtract(np.repeat([[320,240]],4, axis=0), coordinates)

def conv_pix_2_norm(coordinates: np.array, camera_matrix: np.array) -> np.array:
    """
    Use the camera parameters to calculate coordinates of the normalized plane (x,y) from pixel coordinates (u,v)
    """
    _c_u = camera_matrix[0,2]
    _c_v = camera_matrix[1,2]
    _f_x = camera_matrix[0,0]
    _f_y = camera_matrix[1,1]
    if not np.shape(coordinates) == (4,2):
        rospy.ERROR("Array passed to conv_pix_2_norm does not have shape (4,2)")

    coordinates_new = np.subtract(coordinates, np.repeat([[_c_u, _c_v]],4, axis=0))
    coordinates_new = np.divide(coordinates, np.repeat([[_f_x, _f_y]],4, axis=0))
    return coordinates_new

def conv_norm_2_pix(coordinates: np.array, camera_matrix: np.array) -> np.array:
    """
    Use the camera parameters to calculate pixel coordinates (u,v) from the the normalized coordinates (x,y)
    """
    _c_u = camera_matrix[0,2]
    _c_v = camera_matrix[1,2]
    _f_x = camera_matrix[0,0]
    _f_y = camera_matrix[1,1]
    coordinates_new = np.multiply(coordinates, np.repeat([[_f_x, _f_y]],4, axis=0))
    coordinates_new = np.add(coordinates, np.repeat([[_c_u, _c_v]],4, axis=0))
    coordinates_new = coordinates_new.astype(int) # pixel values are always integers
    return coordinates_new

def get_pose_from_coordinates(coordinates: np.array, camera_matrix=None, marker_len=None, camera_distortion=None) -> Pose:
    """
    This function receives the coordinates of 4 corner points and returns their pose estimation respect to the camera
    If the optional args are not given they will be read from the ROS Params
    """
    # initiate geometry_msgs objects
    trans = Point()
    rot = Quaternion()
    pose = Pose()

    try: # importing parameters from ROS if not given as args
        if (np.array(camera_matrix) == None).any():
            _camera_matrix_dict= rospy.get_param("camera_matrix")
            _camera_matrix = np.array(_camera_matrix_dict['data'])
            camera_matrix = np.reshape(_camera_matrix, (_camera_matrix_dict['rows'], _camera_matrix_dict['cols']))
        
        if marker_len == None:
            _marker_len = rospy.get_param("aruco_marker_length")
            marker_len = float(_marker_len)
        
        if (np.array(camera_distortion) == None).any():
            _camera_distortion_dict= rospy.get_param("distortion_coefficients")
            _camera_distortion = np.array(_camera_distortion_dict['data'])
            camera_distortion = np.reshape(_camera_distortion, (_camera_distortion_dict['rows'], _camera_distortion_dict['cols']))
    except:
        rospy.logerr("utils.get_pose_from_coordinates: Could not read ROS Params")
        return pose # empty pose

    # Use the built-in pose estimation of opencv to estimate the pose of the aruco marker found by detect_markers
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(coordinates, marker_len, camera_matrix, camera_distortion)

    # create Point representing translation of estimated pose
    [trans.x, trans.y, trans.z] = tvecs[0][0]

    # create Quaternion representing rotation of estimated pose
    _r = Rotation.from_rotvec(rvecs[0][0]) # Rotation object from SciPy
    [rot.x, rot.y, rot.z, rot.w] = Rotation.as_quat(_r)

    # compose and publish Pose object
    pose.position=trans
    pose.orientation=rot

    return pose

def get_coordinates_from_pose(pose : Pose, camera_matrix=None, marker_len=None) -> np.array:
    """
    This function receives the pose of a marker with respect to the camera
    and returns the coordinates of 4 corner points in the image plane
    If the optional args are not given they will be read from the ROS Params
    """
    # corner points in the marker coordinate system
    hl = marker_len/2
    corners_in_marker=np.array([[hl, hl, 0, 1], [-hl, hl, 0, 1], [-hl, -hl, 0, 1], [hl, -hl, 0, 1]])

    # Create a pytransform corresponding to the passed pose
    marker2cam_pose_array = np.array([pose.position.x, pose.position.y, pose.position.z, 
              pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]) # vector and quaternion concat
    marker2cam = pt.transform_from_pq(marker2cam_pose_array)

    corners_in_cam = pt.transform(A2B=marker2cam, PA=corners_in_marker) # transform points into camera frame
    corners_in_cam_T = np.transpose(corners_in_cam)

    camera_mat_Homog = np.append(camera_matrix, np.zeros((3,1)), axis=1)
    corners_in_img_Homog = np.matmul(camera_mat_Homog, corners_in_cam_T)


    print('corners_in_img_Homog : \n '+str(corners_in_img_Homog))
    corners_in_img = np.transpose(np.array([np.divide(corners_in_img_Homog[0,:],corners_in_img_Homog[2,:]), # divide with scale for each
                                            np.divide(corners_in_img_Homog[1,:],corners_in_img_Homog[2,:])]))
    return corners_in_img