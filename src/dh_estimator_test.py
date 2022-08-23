import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
from dh_estimator import DHestimator,RLS

def plot_param_errors(param_errors_list, axis):
    num_total, len = np.shape(param_errors_list)
    assert num_total%4==0, "number of DH params must be divisible by 4"
    num = num_total//4
    X = range(0,len)
    colors = np.array(['tab:blue','tab:orange', 'tab:green','tab:red','tab:purple','tab:olive','tab:cyan','tab:pink','tab:brown','tab:gray'])
    if (num>colors.size):
        colors = np.random.choice(colors, size=(num,), replace=True, p=None)
    for i in range(num):
        axis[0,0].plot(X,param_errors_list[i,:].flatten(), color=colors[i],   label=str(i))
    axis[0,0].set_title(r'$\Delta$$\theta$')
    axis[0,0].legend()

    for i in range(num):
        axis[0,1].plot(X,param_errors_list[i+8,:].flatten(), color=colors[i],   label=str(i))
    axis[0,1].set_title(r'$\Delta$d')
    axis[0,1].legend()

    for i in range(num):
        axis[1,0].plot(X,param_errors_list[i+16,:].flatten(), color=colors[i],   label=str(i))
    axis[1,0].set_title(r'$\Delta$a')
    axis[1,0].legend()

    for i in range(num):
        axis[1,1].plot(X,param_errors_list[i+24,:].flatten(), color=colors[i],   label=str(i))
    axis[1,1].set_title(r'$\Delta$$\alpha$')
    axis[1,1].legend()

def plot_traj(traj, axis):
    num, len = np.shape(traj)
    X = range(0,len)
    colors = np.array(['tab:blue','tab:orange', 'tab:green','tab:red','tab:purple','tab:olive','tab:cyan','tab:pink','tab:brown','tab:gray'])
    if (num>colors.size):
        colors = np.random.choice(colors, size=(num,), replace=True, p=None)
    for i in range(num):
        axis.plot(X,traj[i,:].flatten(), color=colors[i],   label=str(i))
    axis.set_title(r'$\theta$ trajectories')
    axis.legend()

def plot_curr_est(param_errors_list, axis):
    num_total, len = np.shape(param_errors_list)
    assert num_total%4==0, "number of DH params must be divisible by 4"
    num = num_total//4

    X = [n for n in range(0,num)]

    axis[0,0].clear()
    Y = param_errors_list[0:num,len-1]
    axis[0,0].stem(X,Y)
    axis[0,0].set_title(r'$\Delta$$\theta$')

    axis[0,1].clear()
    Y = param_errors_list[num:2*num,-1]
    axis[0,1].stem(X,Y)
    axis[0,1].set_title(r'$\Delta$d')

    axis[1,0].clear()
    Y = param_errors_list[2*num:3*num,-1]
    axis[1,0].stem(X,Y)
    axis[1,0].set_title(r'$\Delta$a')

    axis[1,1].clear()
    Y = param_errors_list[3*num:4*num,-1]
    axis[1,1].stem(X,Y)
    axis[1,1].set_title(r'$\Delta$$\alpha$')

theta_nom=np.array([0,0,0,0,0,0,0,0,0])
d_nom=np.array([0,0,0.42,0,0.4,0,0,0,0])
a_nom=np.array([0,0,0,0,0,0,0,0,0])
alpha_nom=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2, np.pi/2,0])

assert theta_nom.size == d_nom.size == a_nom.size == alpha_nom.size, "All parameter vectors must have same length"
num_links = theta_nom.size

d_real=d_nom + np.array([0,0,0,0,0,0,0,0.05,0])
a_real=a_nom + np.array([0,0,0,0,0,0,0,0,0])
alpha_real=alpha_nom + np.array([0,0,0,0,0,0,0,0,0]) #0.0002


end=200
estimator=DHestimator()
param_errors_list=np.zeros((4*num_links,end))
jacobian=np.zeros((0,4*num_links))
pos_error=np.zeros((0,1))
traj=np.zeros((num_links,end))

rls=RLS(4*num_links,1)

for k in range(0,end):
    ######## trajectory:
    theta_nom=theta_nom + np.random.default_rng().normal(0, 0.01, (num_links,))
    theta_real=theta_nom + np.array([0,0,0,0,0,0,0,0,0])
    traj[:,k] = np.transpose(theta_real)

   
    jacobian = estimator.get_parameter_jacobian(theta_nom, d_nom, a_nom, alpha_nom)
    T_nom = estimator.get_T__i0(8,theta_nom, d_nom, a_nom, alpha_nom)
    T_real = estimator.get_T__i0(8, theta_real, d_real, a_real, alpha_real)
    nominal_pos = T_nom[0:3,3].reshape((3,1))
    real_pos = T_real[0:3,3].reshape((3,1))

    nom_rot=T_nom[0:3,0:3]
    real_rot=T_real[0:3,0:3]
    delta_x=0.5*(real_rot[2,1]-real_rot[1,2]-nom_rot[2,1]+nom_rot[1,2])
    delta_y=0.5*(real_rot[0,2]-real_rot[2,0]-nom_rot[0,2]+nom_rot[2,0])
    delta_z=0.5*(real_rot[1,0]-real_rot[0,1]-nom_rot[1,0]+nom_rot[0,1])

    current_error=np.concatenate((real_pos-nominal_pos,np.reshape(np.array([delta_x,delta_y,delta_z]),(3,1))),axis=0)

  
    # use RLS
    rls.add_obs(S=jacobian, Y=current_error)
    param_errors_list[:,k] = rls.get_estimate().flatten()


fig1, ax1 = plt.subplots(2, 2)
plot_param_errors(param_errors_list=param_errors_list, axis=ax1)

fig2, ax2 = plt.subplots()
plot_traj(traj=traj, axis=ax2)

fig3, ax3 = plt.subplots(2, 2)
plot_curr_est(param_errors_list=param_errors_list, axis=ax3)

plt.show()
print('fin')
