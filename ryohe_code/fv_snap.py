# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:41:27 2023

@author: 81905
"""

import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.patches as pat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
import os



#==================================================================def function
def bc(dx,dy):
    if dx > LX/2:
        dx=dx-LX
    elif dx < -LX/2:
        dx=dx+LX
    if dy > LY/2:
        dy=dy-LY
    elif dy < -LY/2:
        dy=dy+LY
    return(dx,dy)

#===============================================================calc. parameter
SVC='S'
pi = math.pi
n = 64**2
for i in range(10000000000):
    if i**2 < n:
        pass
    else:
        N_all = i
        break

N = 64**2
for i in range(10000000000):
    if i**2 < n:
        pass
    else:
        N_all = i
        break

#===================各パラメーターの設定←par_occとfile_number注意==================
a = 1.4
pi = np.pi
file_number='1'
par_occ='0790'
pack = float(par_occ)/1000
par_occ1 = str(pack)
Lx = 64
Ly = Lx
LX = Lx
LY = Ly

#==================================================================データセット読み込み
loaded_file='./actual_dataset_updated_ver3/'+par_occ+'/'+file_number+'.npz'  
file=np.load(loaded_file, allow_pickle=True)

"""
sigma1 = math.sqrt(pack*Lx*Ly/N/math.pi)
sigma = [sigma1 for i in range(N)]
table2 = sigma###半径リスト、単成分
"""
#print(table2)

#=====================================================================file read
sigma = file['sigma_list']
table2 = file['sigma_list']
table = file['initial_position']
angle = file['angle']
NN = file['fv_NN']

"""
table = np.loadtxt('N0032_0740_002_01.dat')
#print(table)
with open('angle_0740_first_32.txt', 'rb') as pos_data07:
    angle = pickle.load(pos_data07)
angle = angle[0]###各粒子iに対する第一近傍の粒子NNに対応した各二成分のangle
#print(angle)
with open('NN_0740_first_32.txt', 'rb') as f:
    NN = pickle.load(f)
NN = NN[0]###各粒子iに対する第一近傍の粒子番号のリスト
#print(NN)
"""

#==========================================================================main
fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)
plt.axis([20,30,42,52])
#ax.set_xlim(xmin, xmax)
#ax.set_ylim(ymin, ymax)
#ax.set_xlim(8.28663755911401-3, 8.28663755911401+3)
#ax.set_ylim(9.64372042922696-3, 9.64372042922696+3)

#----粒子の描画
for i in range(N):
    
    xi=table[i][0]###tableのi行0列目はx座標、1列目はy座標
    yi=table[i][1]
    a=table2[i]
    
    col='none'
    c=pat.Circle(xy=(xi,yi), radius=a, color = col, ec='black')
    ax.add_patch(c)
    msg=str(i) #番号付ける
    plt.annotate(msg, xy=(xi, yi))
    
    if xi < a:
        c=pat.Circle(xy=(xi+LX,yi), radius=a, color = col, ec='black')
        ax.add_patch(c)
    elif Lx - xi < a:
        c=pat.Circle(xy=(xi-LX,yi), radius=a, color = col, ec='black')
        ax.add_patch(c)
    if yi < a:
        c=pat.Circle(xy=(xi,yi+LY), radius=a, color = col, ec='black')
        ax.add_patch(c)
    elif Ly - yi < a:
        c=pat.Circle(xy=(xi,yi-LY), radius=a, color = col, ec='black')
        ax.add_patch(c)
    if xi < a and yi < a:
        c=pat.Circle(xy=(xi+LX,yi+LY), radius=a, color = col, ec='black')
        ax.add_patch(c)
    elif Lx - xi < a and yi < a:
        c=pat.Circle(xy=(xi-LX,yi+LY), radius=a, color = col, ec='black')
        ax.add_patch(c)
    elif Ly - xi < a and Ly - yi < a:
        c=pat.Circle(xy=(xi-LX,yi-LY), radius=a, color = col, ec='black')
        ax.add_patch(c)
    elif xi < a and Lx - yi < a:
        c=pat.Circle(xy=(xi+LX,yi-LY), radius=a, color = col, ec='black')
        ax.add_patch(c)

#----自由体積の描画
for i in range(0,N):
    xi=table[i][0]
    yi=table[i][1]
    ri=sigma[i]
    
    for j in range(len(NN[i])):
        xj=table[NN[i][j]][0]
        yj=table[NN[i][j]][1]
        rj=table2[NN[i][j]]
        
        dx,dy=bc(xj-xi,yj-yi)           
        xj=xi+dx
        yj=yi+dy
        th1=math.degrees(angle[i][j][0])
        th2=math.degrees(angle[i][j][1])
        a=ri+rj
        
        c=pat.Arc(xy=(xj,yj),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=1)
        ax.add_patch(c)
        
        if xi >= LX-a:
            c=pat.Arc(xy=(xj-LX,yj),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
            ax.add_patch(c)
            if yi >= LY-a:
                c=pat.Arc(xy=(xj-LX,yj-LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
                ax.add_patch(c)
            elif yi <= a:
                c=pat.Arc(xy=(xj-LX,yj+LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
                ax.add_patch(c)

        elif xi <= a:
            c=pat.Arc(xy=(xj+LX,yj),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
            ax.add_patch(c)
            if yi >= LY-a:
                c=pat.Arc(xy=(xj+LX,yj-LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
                ax.add_patch(c)
            elif yi <= a:
                c=pat.Arc(xy=(xj+LX,yj+LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
                ax.add_patch(c)
        if yi >= LY-a:
            c=pat.Arc(xy=(xj,yj-LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
            ax.add_patch(c)
        elif yi <= a:
            c=pat.Arc(xy=(xj,yj+LY),width=a*2,height=a*2,theta1=th1,theta2=th2,ec='red',linewidth=2)
            ax.add_patch(c)
     
#===============================================画像ファイル保存
#plt.savefig("./fv_snap_shot/"+str(par_occ)+"/"+str(file_number)+"/"+"4091"+".png",dpi=200)
plt.savefig("./fv_snap_shot/"+str(par_occ)+"/"+str(file_number)+"/"+"4091"+".png",dpi=200)

