import rospy
import numpy as np
import math
import matplotlib.pyplot as plt
from dh_estimator import DHestimator,RLS



theta_nom=np.array([0,0,0,0,0,0,0])
d_nom=np.array([0,0,0.42,0,0.4,0,0])
a_nom=np.array([0,0,0,0,0,0,0])
alpha_nom=np.array([0,np.pi/2,-np.pi/2,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2])

# theta_real=theta_nom + np.array([0,0,0,0,0,0,0])
d_real=d_nom + np.array([0,0,0,0,0,0,0.01])
a_real=a_nom + np.array([0,0,0,0,0,0,0.02])
alpha_real=alpha_nom + np.array([0,0,0.002,0,0,0,0]) #0.0002

end=200
estimator=DHestimator()
printvar=np.zeros((1,end))
param_errors_list=np.zeros((28,end))
jacobian=np.zeros((0,28))
pos_error=np.zeros((0,1))
traj=np.zeros((7,end))

rls=RLS(28,1)

for k in range(0,end):
    ######## trajectory:
    theta_nom=theta_nom + np.random.default_rng().normal(0, 0.01, (7,))
    # theta_nom=theta_nom + np.ones(7)*np.pi/(end*2)
    theta_real=theta_nom + np.array([0,0,0,0,0,0,0])
    traj[:,k] = np.transpose(theta_real)

   
    jacobian = estimator.get_parameter_jacobian(theta_nom, d_nom, a_nom, alpha_nom)
    T_nom = estimator.get_T__i0(7,theta_nom, d_nom, a_nom, alpha_nom)
    T_real = estimator.get_T__i0(7, theta_real, d_real, a_real, alpha_real)
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


X = range(0,end)
figure, axis = plt.subplots(2, 2)

axis[0,0].plot(X,param_errors_list[0,:].flatten(), color='tab:blue',   label='0')
axis[0,0].plot(X,param_errors_list[1,:].flatten(), color='tab:orange', label='1')
axis[0,0].plot(X,param_errors_list[2,:].flatten(), color='tab:green',  label='2')
axis[0,0].plot(X,param_errors_list[3,:].flatten(), color='tab:red',    label='3')
axis[0,0].plot(X,param_errors_list[4,:].flatten(), color='tab:purple', label='4')
axis[0,0].plot(X,param_errors_list[5,:].flatten(), color='tab:olive',  label='5')
axis[0,0].plot(X,param_errors_list[6,:].flatten(), color='tab:cyan',   label='6')
axis[0,0].set_title("d theta")
axis[0,0].legend()

axis[0,1].plot(X,param_errors_list[7,:].flatten(), color='tab:blue',   label='0')
axis[0,1].plot(X,param_errors_list[8,:].flatten(), color='tab:orange', label='1')
axis[0,1].plot(X,param_errors_list[9,:].flatten(), color='tab:green',  label='2')
axis[0,1].plot(X,param_errors_list[10,:].flatten(), color='tab:red',    label='3')
axis[0,1].plot(X,param_errors_list[11,:].flatten(), color='tab:purple', label='4')
axis[0,1].plot(X,param_errors_list[12,:].flatten(), color='tab:olive',  label='5')
axis[0,1].plot(X,param_errors_list[13,:].flatten(), color='tab:cyan',   label='6')
axis[0,1].set_title("d d")
axis[0,1].legend()

axis[1,0].plot(X,param_errors_list[14,:].flatten(), color='tab:blue',   label='0')
axis[1,0].plot(X,param_errors_list[15,:].flatten(), color='tab:orange', label='1')
axis[1,0].plot(X,param_errors_list[16,:].flatten(), color='tab:green',  label='2')
axis[1,0].plot(X,param_errors_list[17,:].flatten(), color='tab:red',    label='3')
axis[1,0].plot(X,param_errors_list[18,:].flatten(), color='tab:purple', label='4')
axis[1,0].plot(X,param_errors_list[19,:].flatten(), color='tab:olive',  label='5')
axis[1,0].plot(X,param_errors_list[20,:].flatten(), color='tab:cyan',   label='6')
axis[1,0].set_title("d a")
axis[1,0].legend()

axis[1,1].plot(X,param_errors_list[21,:].flatten(), color='tab:blue',   label='0')
axis[1,1].plot(X,param_errors_list[22,:].flatten(), color='tab:orange', label='1')
axis[1,1].plot(X,param_errors_list[23,:].flatten(), color='tab:green',  label='2')
axis[1,1].plot(X,param_errors_list[24,:].flatten(), color='tab:red',    label='3')
axis[1,1].plot(X,param_errors_list[25,:].flatten(), color='tab:purple', label='4')
axis[1,1].plot(X,param_errors_list[26,:].flatten(), color='tab:olive',  label='5')
axis[1,1].plot(X,param_errors_list[27,:].flatten(), color='tab:cyan',   label='6')
axis[1,1].set_title("d alpha")
axis[1,1].legend()
figure.show()

fig, ax = plt.subplots()
ax.plot(X,traj[0,:].flatten(), color='tab:blue',   label='0')
ax.plot(X,traj[1,:].flatten(), color='tab:orange', label='1')
ax.plot(X,traj[2,:].flatten(), color='tab:green',  label='2')
ax.plot(X,traj[3,:].flatten(), color='tab:red',    label='3')
ax.plot(X,traj[4,:].flatten(), color='tab:purple', label='4')
ax.plot(X,traj[5,:].flatten(), color='tab:olive',  label='5')
ax.plot(X,traj[6,:].flatten(), color='tab:cyan',   label='6')
ax.set_title("theta trajectories")
ax.legend()
fig.show()
plt.show()
print('fin')
