#! /usr/bin/env python3
import math
import numpy as np
from numpy import cos
from numpy import sin
"""
    # {Yuqing Zong}
    # {yzong@kth.se}
"""

def scara_IK(point):
    x = point[0]
    y = point[1]
    z = point[2]
    q = [0.0, 0.0, 0.0]
    l0=0.07
    l1=0.3
    l2=0.35
    #cos_beta=math.cos((l1**2+((x-l0)**2+y**2)-l2**2)/(2*l1*math.sqrt((x-l0)**2+y**2)))
    cosq2 = ((x-l0)**2+y**2-l1**2-l2**2)/(2*l1*l2)
    q1=math.atan2(y, (x-l0))-math.atan2(l2*(math.sqrt(1-cosq2**2)), l1+l2*cosq2)
    q2=math.acos(cosq2)
    q3=z
    q=[q1,q2,q3]
    

    return q

a = 0.311   #global
b = 0.39
c = 0.4
d = 0.078

def kuka_IK(point, R, joint_positions):
    x = point[0]
    y = point[1]
    z = point[2]
    q = np.array(joint_positions)
    X_p = point
    error = 0.01
    E = 1
    while E >= error:
        Jacobian, T_m = creat_Jacobian(q)
        J_pes_inv = np.linalg.pinv(Jacobian)   # creat jacobian and pseudo inverse
        A_m = np.ones((7,4,4))
        A_m[2,:,:] = np.dot(T_m[0, :, :], T_m[1, :, :])
        for i in range(3,7):
            A_m[i,:,:] = np.dot(A_m[i-1,:,:], T_m[i-1, :, :])
        K = np.dot(A_m[6,:,:], T_m[6, :, :])  
        X_h = np.dot(K, [0, 0, d, 1])[0:3] + [0,0,a]
        epsilon_x1 = X_h - X_p
        R = np.array(R)
        epsilon_x2 = 1/2 * (np.cross(R[:,0], K[0:3,0]) + np.cross(R[:,1], K[0:3,1]) + np.cross(R[:,2], K[0:3,2]))   #caculate the epsilon
        epsilon_x = np.concatenate((epsilon_x1, epsilon_x2))
        epsilon_theta = np.dot(J_pes_inv, epsilon_x)
        E = np.linalg.norm(epsilon_theta)
        q = q - epsilon_theta

    return q
    
def creat_Jacobian(joint_positions):  #get the jacobian
    Z_m = np.ones((3, 8))    
    t_m = np.ones((3, 8))
    Jacobian = np.ones((6, 7))
    Z_m[: , 0] = [0, 0, 1] 
    t_m[: , 0] = [0, 0, 0] 
    q1,q2,q3,q4,q5,q6,q7 = joint_positions
    Table = np.array([[math.pi/2, 0, 0, q1], 
          [-math.pi/2, 0, 0, q2], 
          [-math.pi/2, c, 0, q3], 
          [math.pi/2, 0, 0, q4], 
          [math.pi/2, b, 0, q5], 
          [-math.pi/2, 0, 0, q6], 
          [0, 0, 0, q7]])
    Table_0 = Table[:,0]
    Table_1 = Table[:,1]
    Table_2 = Table[:,2]
    Table_3 = Table[:,3]
    T_m = creat_T_m(Table_0,Table_1,Table_2,Table_3)
 
    A_m = creat_A_m(T_m)

    for i in range(0, 7):
        Z_m[:, i+1] = A_m[i, :, :][0: 3, 2]    
        t_m[:, i+1] = A_m[i, :, :][0: 3, 3]   
        Jacobian[:,i]=np.concatenate((np.cross(np.array(Z_m[:, i]), (np.array(t_m[:, 7]) - np.array(t_m[:, i]))), np.array(Z_m[:, i])))  
    return Jacobian, T_m

    
def creat_T_m(Table_0,Table_1,Table_2,Table_3):
    T_m = np.ones((7, 4, 4))
    for i in range(0,7):
        T_m[i,:,:] = creat_Tmatrix(Table_0[i], Table_1[i], Table_2[i], Table_3[i])  

    return T_m

def creat_A_m(T_m):
    A_m = np.ones((7, 4, 4))  
    A_m[0, :, :] = T_m[0, :, :]
    for i in range(1, 7):
        A_m[i, :, :] = np.dot(A_m[i-1, :, :], T_m[i, :, :])

    return A_m

    
def creat_Tmatrix(alpha, d, a, theta): 
    T_m = np.ones((4,4))
    T_m[:,0] = [math.cos(theta),math.sin(theta),0,0]
    T_m[:,1] = [-math.cos(alpha)*math.sin(theta),math.cos(alpha)*math.cos(theta),math.sin(alpha),0]
    T_m[:,2] = [math.sin(alpha)*math.sin(theta),-math.sin(alpha)*math.cos(theta), math.cos(alpha),0]
    T_m[:,3] = [a*math.cos(theta),a*math.sin(theta),d,1]

    return T_m


