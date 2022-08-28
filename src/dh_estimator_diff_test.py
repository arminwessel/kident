import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
from dh_estimator import DHestimator,RLS
import cv2

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
        axis[0,1].plot(X,param_errors_list[i+7,:].flatten(), color=colors[i],   label=str(i))
    axis[0,1].set_title(r'$\Delta$d')
    axis[0,1].legend()

    for i in range(num):
        axis[1,0].plot(X,param_errors_list[i+14,:].flatten(), color=colors[i],   label=str(i))
    axis[1,0].set_title(r'$\Delta$a')
    axis[1,0].legend()

    for i in range(num):
        axis[1,1].plot(X,param_errors_list[i+21,:].flatten(), color=colors[i],   label=str(i))
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

theta_nom=np.array([0,np.pi,np.pi,0,np.pi/2,0,np.pi/2])
d_nom=np.array([0.1525,0.2025,0.2325,0.1825,0.2175,0.1825,0.081])
a_nom=np.array([0,0,0,0,0,0,0])
alpha_nom=np.array([0,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,np.pi/2,-np.pi/2])

assert theta_nom.size == d_nom.size == a_nom.size == alpha_nom.size, "All parameter vectors must have same length"
num_links = theta_nom.size

# testerfahrungen
d_real=d_nom + np.array([0,0,0,0,0,0,0]) # auf d0 und d1 scheint es sich nicht auszuwirken, rest getestet mit 0.05
a_real=a_nom + np.array([0,0,0,0,0,0,0]) # mit 0.0002 funktionieren alle
alpha_real=alpha_nom + np.array([0,0,0,0,0,0,0.002]) # 0.0002 fürhrt bei al0,al1,al2 zu etwa 0.00015, bei al3 führt 0.002 auf 0.001, 
# al4 kommt mit 0.002 auf knapp 0.0008, al5 mit 0.002 auf 0.0015, al6 wia l4

end=200
estimator=DHestimator()
param_errors_list=np.zeros((4*num_links,end))
jacobian=np.zeros((0,4*num_links))
pos_error=np.zeros((0,1))
traj=np.zeros((num_links,end))
distances=np.zeros((0,))

rls=RLS(4*num_links,1)

for k in range(0,2*end):
    # k
    theta_nom=theta_nom + np.random.default_rng().normal(0, 0.01, (num_links,))
    theta_real=theta_nom + np.array([0,0,0,0,0,0,0])
    traj[:,k//2] = np.transpose(theta_real)

   
    jacobian1 = estimator.get_parameter_jacobian(theta_nom, d_nom, a_nom, alpha_nom)
    T_nom1 = estimator.get_T__i0(num_links,theta_nom, d_nom, a_nom, alpha_nom)
    T_real1 = estimator.get_T__i0(num_links, theta_real, d_real, a_real, alpha_real)
    nominal_pos1 = T_nom1[0:3,3].reshape((3,1))
    real_pos1 = T_real1[0:3,3].reshape((3,1))
    rvec1real = cv2.Rodrigues(T_real1[0:3,0:3])[0]
    rvec1nom = cv2.Rodrigues(T_nom1[0:3,0:3])[0]

    theta_nom=theta_nom + np.random.default_rng().normal(0, 0.09, (num_links,))
    theta_real=theta_nom + np.array([0,0,0,0,0,0,0])

    try:
        traj[:,k//2+1] = np.transpose(theta_real)
    except:
        pass
    jacobian2 = estimator.get_parameter_jacobian(theta_nom, d_nom, a_nom, alpha_nom)
    T_nom2 = estimator.get_T__i0(num_links,theta_nom, d_nom, a_nom, alpha_nom)
    T_real2 = estimator.get_T__i0(num_links, theta_real, d_real, a_real, alpha_real)
    nominal_pos2 = T_nom2[0:3,3].reshape((3,1))
    real_pos2 = T_real2[0:3,3].reshape((3,1))
    rvec2real = cv2.Rodrigues(T_real2[0:3,0:3])[0]
    rvec2nom = cv2.Rodrigues(T_nom2[0:3,0:3])[0]

    # nominal difference:
    dtvec_nom = nominal_pos1 - nominal_pos2
    drvec_nom = rvec1nom - rvec2nom
    # real difference:
    dtvec_real = real_pos1 - real_pos2
    drvec_real = rvec1real - rvec2real

    distances = np.append(distances, np.linalg.norm(dtvec_nom))
    
    current_error=np.concatenate((dtvec_real-dtvec_nom,drvec_real-drvec_nom),axis=0)
    jacobian = jacobian1-jacobian2
    # use RLS
    rls.add_obs(S=jacobian, Y=current_error)
    param_errors_list[:,k//2] = rls.get_estimate().flatten()


fig1, ax1 = plt.subplots(2, 2)
plot_param_errors(param_errors_list=param_errors_list, axis=ax1)

fig2, ax2 = plt.subplots()
plot_traj(traj=traj, axis=ax2)

fig3, ax3 = plt.subplots(2, 2)
plot_curr_est(param_errors_list=param_errors_list, axis=ax3)

plt.show()
avg_dist=np.mean(distances)
print('fin')
