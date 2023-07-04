import sys
import os
sys.path.append('/home/isobelab2022/build3/hoomd')
import itertools
import math

import gsd.hoomd
import hoomd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from numba import jit, f8, i8, b1, void,njit
import matplotlib.patches as pat
import sys
import os
import fresnel
from PIL import Image
import IPython
import packaging.version
import random
import time
import pickle


nu=float(sys.argv[1])
fixed_percent=float(sys.argv[2])
kbT=float(sys.argv[3])
L=64
 

media_dir="/media/isobelab2022/data/active_glass/ver7"



ver=str(nu)+"_"+str(fixed_percent)+"_"+str(kbT)
main_dir="./"+ver


traj_dir=media_dir+"/"+ver
traj_path=traj_dir+"/log_pos_"+ver+".gsd"
traj = gsd.hoomd.open(traj_path, 'rb')

data_path=media_dir+"/"+ver+"/data.pickle"
with open(data_path, mode='rb') as f:
    data = pickle.load(f)



dt=data["dt"]
pos_out_steps_period=data["pos_out_steps_period"]
small_sigma=data["sigma_ss"]
msd_times=[i*dt*pos_out_steps_period for i in range(len(traj)-1)]
zerod_time=msd_times/small_sigma

zerod_time=np.array(zerod_time)





all_pos=[traj[i].particles.position for i in range(len(traj)-1)]


all_pos=np.array(all_pos)
# posは（時間、粒子数、3次元[x,y,z]）の次元になっている。

print(len(all_pos))

print(all_pos.shape)

# posの次元を（3次元、粒子数、時間）に変換する。
all_pos=np.transpose(all_pos,(2,1,0))

print(all_pos.shape)


# 次元は（粒子数、時間）
rx=all_pos[0]
ry=all_pos[1]
# rz=pos[2]

# 境界条件を考慮して、粒子の位置を調整する。
def adjust_periodic(x,L):
    for t in range(len(x)-1):
        if x[t+1] - x[t] > L/2:
            x[t+1] -= (x[t+1] - x[t]+L/2)//L*L
        if x[t+1] - x[t] < -L/2:
            x[t+1] += (x[t] - x[t+1]+L/2)//L*L

print(rx.shape)

# (N,T)=rx.shape

for i in range(len(rx)):
    adjust_periodic(rx[i],64)
    adjust_periodic(ry[i],64)
    # adjust_periodic(rz[i],64)


N = rx.shape[0]
T=rx.shape[1]

msd_list = []
ngp_list = []
for t in range(T):
    m2=0.0
    m4=0.0

    for i in range(N):
        dx=rx[i][t]-rx[i][0]
        dy=ry[i][t]-ry[i][0]


        m2_element=dx*dx+dy*dy
        m4_element=m2_element*m2_element
        m2 += m2_element
        m4 += m4_element
        
            
    m2 /= N
    m4 /= N

    msd = m2 / ((small_sigma) ** 2)
# 初回だけngp=0とする
    if (t==0):
        ngp=0
    else:
        ngp = ((m4/(m2**2))/2) - 1 

    msd_list.append(msd)
    ngp_list.append(ngp)


msd_list=np.array(msd_list)
np.savez(traj_dir+"/msd.npz",msd_list=msd_list,zerod_time=zerod_time)


msd_list=np.array(ngp_list)
np.savez(traj_dir+"/ngp.npz",ngp_list=ngp_list,zerod_time=zerod_time)

print("FiNISH")

del traj