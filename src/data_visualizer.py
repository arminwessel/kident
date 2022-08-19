#!/usr/bin/env python3

import json
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kident.msg import Est
import numpy as np
from sensor_msgs.msg import Image
import ros_numpy
import time
from pathlib import Path
import pandas as pd 


class DataVisualizer():
    """
    Store and visualize data
    """
    def __init__(self) -> None:
        self.sub_meas = rospy.Subscriber("est", Est, self.update_plot_data)

        self.len = 20000
        self.param_errors_list=np.empty((28,self.len))
        self.traj=np.empty((7,self.len))
        self.param_errors_list[:] = np.NaN
        self.traj[:] = np.NaN
        self.k = 0

        self.fig_est, self.ax_est = plt.subplots(2,2)
        self.fig_est.set_size_inches(16, 9, forward=True)
        self.fig_est.tight_layout(pad=2)


        self.fig_traj, self.ax_traj = plt.subplots(1,1)
        #nself.ax_traj.set_xlim(0, self.len)
        
        self.fig_curr_est, self.ax_curr_est = plt.subplots(2,2)

        self.pub_plot_est = rospy.Publisher("plot_estimated_params", Image, queue_size=20)
        self.pub_plot_traj = rospy.Publisher("plot_robot_trajectory", Image, queue_size=20)

        ani_est = animation.FuncAnimation(self.fig_est, self.plot_est, interval=20)
        ani_traj = animation.FuncAnimation(self.fig_traj, self.plot_traj, interval=20)
        ani_curr_est = animation.FuncAnimation(self.fig_curr_est, self.plot_curr_est, interval=20)
        plt.show()

    def update_plot_data(self, data):
        estimate_k, traj_k = data.estimate, data.joints
        
        estimate_k = np.array(estimate_k).flatten()
        traj_k = np.array(traj_k).flatten()

        self.param_errors_list[:,self.k] = estimate_k
        self.traj[:,self.k] = traj_k
        self.k += 1
        if (self.k >= self.len):         # reset k to overwrite old data
            self.k=0
            self.save_data_shutdown()
            self.param_errors_list[:] = np.NaN
            self.traj[:] = np.NaN
            

    def plot_est(self, i=None):
        num_total, len = np.shape(self.param_errors_list)
        assert num_total%4==0, "number of DH params must be divisible by 4"
        num = num_total//4

        X = range(len(self.param_errors_list[0,:]))

        colors = np.array([ 'tab:blue','tab:orange','tab:green',
                            'tab:red', 'tab:purple','tab:olive',
                            'tab:cyan','tab:pink',  'tab:brown','tab:gray'])
        if (num>colors.size):
            colors = np.random.choice(colors, size=(num,), replace=True, p=None)

        axis=self.ax_est

        axis[0,0].clear()
        for i in range(num):
            axis[0,0].plot(X,self.param_errors_list[i,:].flatten(), color=colors[i],   label=str(i))
        axis[0,0].set_title(r'$\Delta$$\theta$')
        axis[0,0].legend()

        axis[0,1].clear()
        for i in range(num):
            axis[0,1].plot(X,self.param_errors_list[i+num,:].flatten(), color=colors[i],   label=str(i))
        axis[0,1].set_title(r'$\Delta$d')
        axis[0,1].legend()

        axis[1,0].clear()
        for i in range(num):
            axis[1,0].plot(X,self.param_errors_list[i+2*num,:].flatten(), color=colors[i],   label=str(i))
        axis[1,0].set_title(r'$\Delta$a')
        axis[1,0].legend()

        axis[1,1].clear()
        for i in range(num):
            axis[1,1].plot(X,self.param_errors_list[i+3*num,:].flatten(), color=colors[i],   label=str(i))
        axis[1,1].set_title(r'$\Delta$$\alpha$')
        axis[1,1].legend()

    def plot_traj(self, i=None):
        num, len = np.shape(self.traj)
        X = range(len(self.param_errors_list[0,:]))
        colors = np.array([ 'tab:blue','tab:orange','tab:green',
                            'tab:red', 'tab:purple','tab:olive',
                            'tab:cyan','tab:pink',  'tab:brown','tab:gray'])
        if (num>colors.size):
            colors = np.random.choice(colors, size=(num,), replace=True, p=None)
        axis=self.ax_traj
        axis.clear()
        for i in range(num-1): # num - 1 is used to avoid plotting zero for camera intrinsics param theta
            axis.plot(X,self.traj[i,:].flatten(), color=colors[i],   label=str(i))
        axis.set_title(r'$\theta$ trajectories')
        axis.legend()

    def plot_curr_est(self, i=None):

        self.ax_curr_est[0,0].clear()
        X = [n for n in range(0,7)]
        Y = self.param_errors_list[0:7,self.k-1]
        self.ax_curr_est[0,0].stem(X,Y)
        self.ax_curr_est[0,0].set_title(r'$\Delta$$\theta$')

        self.ax_curr_est[0,1].clear()
        X = [n for n in range(7,14)]
        Y = self.param_errors_list[7:14,self.k-1]
        self.ax_curr_est[0,1].stem(X,Y)
        self.ax_curr_est[0,1].set_title(r'$\Delta$d')

        self.ax_curr_est[1,0].clear()
        X = [n for n in range(14,21)]
        Y = self.param_errors_list[14:21,self.k-1]
        self.ax_curr_est[1,0].stem(X,Y)
        self.ax_curr_est[1,0].set_title(r'$\Delta$a')

        self.ax_curr_est[1,1].clear()
        X = [n for n in range(21,28)]
        Y = self.param_errors_list[21:28,self.k-1]
        self.ax_curr_est[1,1].stem(X,Y)
        self.ax_curr_est[1,1].set_title(r'$\Delta$$\alpha$')

    def plot_curr_est(self, i=None):
        num_total, len = np.shape(self.param_errors_list)
        assert num_total%4==0, "number of DH params must be divisible by 4"
        num = num_total//4      # index is type integer

        X = [n for n in range(0,num)]
        axis=self.ax_curr_est

        axis[0,0].clear()
        Y = self.param_errors_list[0:num,len-1]
        axis[0,0].stem(X,Y)
        axis[0,0].set_title(r'$\Delta$$\theta$')

        axis[0,1].clear()
        Y = self.param_errors_list[num:2*num,-1]
        axis[0,1].stem(X,Y)
        axis[0,1].set_title(r'$\Delta$d')

        axis[1,0].clear()
        Y = self.param_errors_list[2*num:3*num,-1]
        axis[1,0].stem(X,Y)
        axis[1,0].set_title(r'$\Delta$a')

        axis[1,1].clear()
        Y = self.param_errors_list[3*num:4*num,-1]
        axis[1,1].stem(X,Y)
        axis[1,1].set_title(r'$\Delta$$\alpha$')



    def publish_est(self):
        self.plot_est()
        self.fig_est.canvas.draw()
        image_from_plot = np.frombuffer(self.fig_est.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig_est.canvas.get_width_height()[::-1] + (3,))
        self.pub_plot_est.publish(ros_numpy.msgify(Image, image_from_plot.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS


    def publish_traj(self):
        self.plot_traj()
        self.fig_traj.canvas.draw()
        image_from_plot = np.frombuffer(self.fig_traj.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fig_traj.canvas.get_width_height()[::-1] + (3,))
        self.pub_plot_traj.publish(ros_numpy.msgify(Image, image_from_plot.astype(np.uint8), encoding='rgb8')) # convert opencv image to ROS

    def save_data_shutdown(self):
        currtime = time.time()
        Path.cwd().joinpath('saved_plots/img').mkdir(parents=True, exist_ok=True)
        Path.cwd().joinpath('saved_plots/csv').mkdir(parents=True, exist_ok=True)

        savefile_name = str(Path.cwd().joinpath('saved_plots/img', 'estimate_{}.png'.format(int(currtime))))
        self.plot_est()
        self.fig_est.canvas.draw()
        self.fig_est.savefig(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/csv', 'estimate_{}.csv'.format(int(currtime))))
        pd.DataFrame(self.param_errors_list).to_csv(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/img', 'traj_{}.png'.format(int(currtime))))
        self.plot_traj()
        self.fig_traj.canvas.draw()
        self.fig_traj.savefig(savefile_name)

        savefile_name = str(Path.cwd().joinpath('saved_plots/csv', 'traj_{}.csv'.format(int(currtime))))
        pd.DataFrame(self.traj).to_csv(savefile_name)
        rospy.loginfo("Saved data")

    
# Main function.
if __name__ == "__main__":
    rospy.init_node('data_visualizer')   # init ROS node named aruco_detector
    rospy.loginfo('#Node data_visualizer running#')
    while not rospy.get_rostime():      # wait for ros time service
        pass

    dv = DataVisualizer()          # create instance
    rospy.on_shutdown(dv.save_data_shutdown)
    
    rate = rospy.Rate(30) # ROS Rate at ... Hz

    
    
    while not rospy.is_shutdown():
        try:
            rate.sleep()
        except:
            pass

    

    
    

