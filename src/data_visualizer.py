#!/usr/bin/env python3

import json
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kident.msg import Est
import numpy as np
from sensor_msgs.msg import Image
import ros_numpy

class DataVisualizer():
    """
    Store and visualize data
    """
    def __init__(self) -> None:
        self.sub_meas = rospy.Subscriber("est", Est, self.update_plot_data)

        len = 10000
        self.param_errors_list=np.empty((28,len))
        self.traj=np.empty((7,len))
        self.param_errors_list[:] = np.NaN
        self.traj[:] = np.NaN
        self.k = 0

        self.fig_est, self.ax_est = plt.subplots(2,2)
        self.ax_est.axis('off')
        self.fig_est.tight_layout(pad=0)
        self.ax_est.margins(0)

        self.fig_traj, self.ax_traj = plt.subplots(1,1)

        self.pub_plot_est = rospy.Publisher("estimated params", Image, queue_size=20)

    def update_plot_data(self, data):
        estimate_k, traj_k = data.estimate, data.joints
        
        estimate_k = np.array(estimate_k).flatten()
        traj_k = np.array(traj_k).flatten()

        self.param_errors_list[:,self.k] = estimate_k
        self.traj[:,self.k] = traj_k
        self.k += 1
        if (self.k == len):         # reset k to overwrite old data
            self.k=0

    def animate_est(self, i):
        X = range(len(self.param_errors_list[0,:]))
        self.ax_est[0,0].clear()
        self.ax_est[0,0].plot(X,self.param_errors_list[0,:].flatten(), color='tab:blue',   label='0')
        self.ax_est[0,0].plot(X,self.param_errors_list[1,:].flatten(), color='tab:orange', label='1')
        self.ax_est[0,0].plot(X,self.param_errors_list[2,:].flatten(), color='tab:green',  label='2')
        self.ax_est[0,0].plot(X,self.param_errors_list[3,:].flatten(), color='tab:red',    label='3')
        self.ax_est[0,0].plot(X,self.param_errors_list[4,:].flatten(), color='tab:purple', label='4')
        self.ax_est[0,0].plot(X,self.param_errors_list[5,:].flatten(), color='tab:olive',  label='5')
        self.ax_est[0,0].plot(X,self.param_errors_list[6,:].flatten(), color='tab:cyan',   label='6')
        self.ax_est[0,0].set_title("d theta")
        self.ax_est[0,0].legend()

        self.ax_est[0,1].clear()
        self.ax_est[0,1].plot(X,self.param_errors_list[7,:].flatten(), color='tab:blue',   label='0')
        self.ax_est[0,1].plot(X,self.param_errors_list[8,:].flatten(), color='tab:orange', label='1')
        self.ax_est[0,1].plot(X,self.param_errors_list[9,:].flatten(), color='tab:green',  label='2')
        self.ax_est[0,1].plot(X,self.param_errors_list[10,:].flatten(), color='tab:red',    label='3')
        self.ax_est[0,1].plot(X,self.param_errors_list[11,:].flatten(), color='tab:purple', label='4')
        self.ax_est[0,1].plot(X,self.param_errors_list[12,:].flatten(), color='tab:olive',  label='5')
        self.ax_est[0,1].plot(X,self.param_errors_list[13,:].flatten(), color='tab:cyan',   label='6')
        self.ax_est[0,1].set_title("d d")
        self.ax_est[0,1].legend()

        self.ax_est[1,0].clear()
        self.ax_est[1,0].plot(X,self.param_errors_list[14,:].flatten(), color='tab:blue',   label='0')
        self.ax_est[1,0].plot(X,self.param_errors_list[15,:].flatten(), color='tab:orange', label='1')
        self.ax_est[1,0].plot(X,self.param_errors_list[16,:].flatten(), color='tab:green',  label='2')
        self.ax_est[1,0].plot(X,self.param_errors_list[17,:].flatten(), color='tab:red',    label='3')
        self.ax_est[1,0].plot(X,self.param_errors_list[18,:].flatten(), color='tab:purple', label='4')
        self.ax_est[1,0].plot(X,self.param_errors_list[19,:].flatten(), color='tab:olive',  label='5')
        self.ax_est[1,0].plot(X,self.param_errors_list[20,:].flatten(), color='tab:cyan',   label='6')
        self.ax_est[1,0].set_title("d a")
        self.ax_est[1,0].legend()

        self.ax_est[1,1].clear()
        self.ax_est[1,1].plot(X,self.param_errors_list[21,:].flatten(), color='tab:blue',   label='0')
        self.ax_est[1,1].plot(X,self.param_errors_list[22,:].flatten(), color='tab:orange', label='1')
        self.ax_est[1,1].plot(X,self.param_errors_list[23,:].flatten(), color='tab:green',  label='2')
        self.ax_est[1,1].plot(X,self.param_errors_list[24,:].flatten(), color='tab:red',    label='3')
        self.ax_est[1,1].plot(X,self.param_errors_list[25,:].flatten(), color='tab:purple', label='4')
        self.ax_est[1,1].plot(X,self.param_errors_list[26,:].flatten(), color='tab:olive',  label='5')
        self.ax_est[1,1].plot(X,self.param_errors_list[27,:].flatten(), color='tab:cyan',   label='6')
        self.ax_est[1,1].set_title("d alpha")
        self.ax_est[1,1].legend()
        self.fig_est.canvas.draw()
        image_from_plot = np.frombuffer(self.fig_est.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig_est.canvas.get_width_height()[::-1] + (3,))
        self.pub_plot_est.publish(ros_numpy.msgify(Image, image_from_plot.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS


    def animate_traj(self, i):
        X = range(len(self.param_errors_list[0,:]))

        self.ax_traj.clear()
        self.ax_traj.plot(X,self.traj[0,:].flatten(), color='tab:blue',   label='0')
        self.ax_traj.plot(X,self.traj[1,:].flatten(), color='tab:orange', label='1')
        self.ax_traj.plot(X,self.traj[2,:].flatten(), color='tab:green',  label='2')
        self.ax_traj.plot(X,self.traj[3,:].flatten(), color='tab:red',    label='3')
        self.ax_traj.plot(X,self.traj[4,:].flatten(), color='tab:purple', label='4')
        self.ax_traj.plot(X,self.traj[5,:].flatten(), color='tab:olive',  label='5')
        self.ax_traj.plot(X,self.traj[6,:].flatten(), color='tab:cyan',   label='6')
        self.ax_traj.set_title("theta trajectories")
        self.ax_traj.legend()

    
# Main function.
if __name__ == "__main__":
    rospy.init_node('data_visualizer')   # init ROS node named aruco_detector
    rospy.loginfo('#Node data_visualizer running#')

    while not rospy.get_rostime():      # wait for ros time service
        pass

    dv = DataVisualizer()          # create instance
    ani_est = animation.FuncAnimation(dv.fig_est, dv.animate_est, interval=1000)
    ani_traj = animation.FuncAnimation(dv.fig_traj, dv.animate_traj, interval=1000)
    plt.show()
    
    rospy.spin()

    
    

