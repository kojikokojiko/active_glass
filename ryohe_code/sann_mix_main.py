# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:34:21 2022

@author: 81905
"""

########   2nd NN   ########

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import time
import sys
from pathlib import Path

SVC='S'
par_occ = '0800'
file_num = 10
file_num = str(file_num)

N=4096
#N=41
#NNmax=50
NNmax=80
LX=64.0
LY=64.0
#SANN_rcut=1.4*1000
RorA='AVE'
#RorA='RAW'
frame0=1 #4096×300のデータセットで何かの分布を計算
skip=N*frame0
#pass0 = Path('/Users/sotsuken/Desktop/研究/data/bi/N0064_'+par_occ+'_RAW.dat0')
#table=np.loadtxt(pass0,usecols=[0,1,2])

#pass0 = Path('/Users/sotsuken/Desktop/研究/data/bi/kaki/dis1_'+par_occ+'.dat')
#pass0 = Path('20201231_7881_5.0e-6/pos'+par_occ+'.dat')

"""
pass0 = Path('data/pos'+par_occ+'.dat')
table0=np.loadtxt(pass0,usecols=[0,1])
table0=np.delete(table0,slice(skip,None),0)
pass0 = Path('/Users/sotsuken/Desktop/研究/data/bi/N0064_'+par_occ+'_RAW.dat0')
table2=np.loadtxt(pass0,usecols=[2,3])
frame=0
table=np.delete(table0,slice(N*(frame+1),skip),0)
table=np.delete(table,slice(0,N*frame),0)
table=np.append(table,table2,axis=1)
table=np.delete(table,3,axis=1)
"""

#半径の情報を3列目に結合
config = './dataset/0800/initial_pos/0800_'+file_num+'.dat'
table = np.loadtxt(config)

"""
big=0
for i in range(4):
    small=table[i][2]
    if big<small:
        big=small
SANN_rcut=7*big
"""
#半径リストの作成とtableへの結合
Mol_small, Mol_large = 2, 1 ###モル比
Sigma_ratio = 1.4
Density = 0.80     ###占有率
senyuritu = '080'  ###出力
Elast = 1.0     ###反発係数
Sigma_small = np.sqrt(Density*LX*LY*(Mol_large+Mol_small)/N/np.pi/(Mol_large*Sigma_ratio**2 + Mol_small))   ###粒子（小）半径
Sigma_large = Sigma_ratio * Sigma_small

sigma_list = np.full(N, Sigma_large)
sigma_decision = []
with open('sigma_init' + '.txt') as f:
    for line in f:
        sigma_decision_element = float(line)
        sigma_decision.append(sigma_decision_element)
for i in range(N):
    if sigma_decision[i] < 0.1:
        sigma_list[i] = Sigma_small
sigma_list = sigma_list.reshape((4096,1))

#tableの結合        
table=np.concatenate([table, sigma_list], axis=1)
        

#第3近傍までの候補粒子特定の範囲（慎重に決める）

###占有率0.78の場合

#SANN_rcut=0.6*2*3
#SANN_rcut=0.65*2*3
#SANN_rcut=0.7*2*3 #←占有率0.78ではこの値でもいいが
#SANN_rcut=0.75*2*3 #←占有率0.78でもこの値の方が安全

###占有率0.79の場合

#SANN_rcut=0.5*2*3
#SANN_rcut=0.6*2*3
#SANN_rcut=0.65*2*3
#SANN_rcut=0.7*2*3 #←占有率0.79ではこの値？
#SANN_rcut=0.75*2*3 #←占有率0.79でもこの値の方が安全

###占有率0.80の場合
#SANN_rcut=0.6*2*3
#SANN_rcut=0.65*2*3
#SANN_rcut=0.68*2*3
#SANN_rcut=0.7*2*3 
SANN_rcut=0.75*2*3 #←占有率0.80ではこの値が安全

###### For efficiency #######
### grid
'''
if par_occ == 0.72:gd_x=116 # nu=0.720
elif par_occ == 0.76: gd_x=112 # nu=0.760
elif par_occ == 0.78:gd_x=111 # nu=0.780
else :gd_x=1
gd_y=gd_x

### boundary
bc_x=[0 for i in range(-gd_x,gd_x+gd_x)]
bc_y=[0 for i in range(-gd_y,gd_y+gd_y)]
def bc_2d(bc_x,bc_y):
  for i in range(0,gd_x):
    bc_x[i]=i
    bc_x[-i-1]=gd_x-i-1
    bc_x[gd_x+i]=i
  for i in range(0,gd_y):
    bc_y[i]=i
    bc_y[-i-1]=gd_y-i-1
    bc_y[gd_y+i]=i
 
#bc_2d(bc_x,bc_y)

### list vector
ls_x=[-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0]
ls_y=[-5,-4,-3,-2,-1,0,1,2,3,4,5,-5,-4,-3,-2,-1,0,1,2,3,4,5,-5,-4,-3,-2,-1,0,1,2,3,4,5,-5,-4,-3,-2,-1,0,1,2,3,4,5,-5,-4,-3,-2,-1,0,1,2,3,4,5,-5,-4,-3,-2,-1]

start=time.time() #2.7364799976348877

NNnum2_5=0
NNnum2_6=0
NNnum2_7=0
NNnum2_8=0
NNnum2_9=0
NNnum2_10=0
NNnum2_11=0
NNnum2_12=0
NNnum2_13=0
NNnum2_14=0
NNnum2_15=0
NNnum2_16=0
NNnum2_17=0

### grid mapping
gd_map=[[-1 for i in range(0,gd_x)] for j in range(0,gd_y)]
gd_mx=[0 for i in range(N)]
gd_my=[0 for i in range(N)]
def gd_mp(gd_map,gd_mx,gd_my):
  for i in range(0,N):
    x=table[i][0]
    y=table[i][1]
    gx=int(x*gd_x/LX)
    gy=int(y*gd_y/LY)
    gd_map[gx][gy]=i
    gd_mx[i]=gx
    gd_my[i]=gy

#gd_mp(gd_map,gd_mx,gd_my)

ls_gd=[[-1 for i in range(N)] for k in range(121)]
def ls3_grid(ls_gd):
  for i in range(0,N):
      #    k=1
    for m in range(0,60):
      j=gd_map[bc_x[gd_mx[i]+ls_x[m]]][bc_y[gd_my[i]+ls_y[m]]]
      if j != -1:
        ls_gd[m][i]=j
#        ls_gd[119-m][j]=i #いらない？
    #        k=k+1
#    ls_gd[49][i]=k-1
#    for m in range(25,48):
#      j=gd_map[bc_x[gd_mx[i]+ls_x[m]]][bc_y[gd_my[i]+ls_y[m]]]
#      if j != 0:
#        ls_gd[k][i]=j
#        k=k+1
#    ls_gd[50][i]=k-1

#ls3_grid(ls_gd)
'''


##### MAIN ROUTINE (SANN) #####

### 1st NN ###
NNnum=[0 for i in range(N)]
NNnum2=[0 for i in range(N)]

NN=[[-1 for i in range(NNmax)] for j in range(N)]
NN1=[[-1 for i in range(10)] for j in range(N)]
NN2=[[-1 for i in range(20)] for j in range(N)]
NN3=[[-1 for i in range(32)] for j in range(N)]

NNnum_S=[0 for i in range(N)]
NN_S=[[0 for i in range(NNmax)] for j in range(N)]
DR_S=[[0 for i in range(NNmax)] for j in range(N)]


for i in range(0,N):
#  k=0
  xi=table[i][0]
  yi=table[i][1]
  print(i)
  for m in range(i+1,N):
      xj=table[m][0]
      yj=table[m][1]
      dxij=xi-xj
      dyij=yi-yj
#periodic boundary conditions
      if dxij > LX/2:
        dxij  = dxij - LX
      elif dxij < -LX/2:
        dxij = dxij + LX
      if dyij > LY/2:
        dyij = dyij - LY
      elif dyij < -LY/2:
        dyij = dyij + LY
#candidates for N.N.
      dr=np.sqrt(dxij**2+dyij**2)
      if dr < SANN_rcut:
        NN_S[i][NNnum_S[i]]=m
        DR_S[i][NNnum_S[i]]=dr
        NN_S[m][NNnum_S[m]]=i
        DR_S[m][NNnum_S[m]]=dr
        NNnum_S[i]+=1
        NNnum_S[m]+=1
#        NNnum_S[i]=NNnum_S[i]+1
#        NN_S[i][k]=j
#        DR_S[i][k]=dr
#        k=k+1
#for i in range(N):
#    NNnum_S[i]=NNnum_S[i]-1

### 2nd step --- sort in increasing order of distance
for i in range(N):
  NNs_list=[[DR_S[i][k],NN_S[i][k]] for k in range(NNnum_S[i])]
  NNs_list.sort(key=lambda x:x[0])
  for k in range(NNnum_S[i]):
    DR_S[i][k]=NNs_list[k][0]
    NN_S[i][k]=NNs_list[k][1]



### 3rd step --- 1st NN --- 2D SANN detection of NN by bisection method
print('1stNN開始')
r_sann_m=[0 for i in range(N)]

Nmax=100
for i in range(N):
  for j in range(3,NNnum_S[i]):
    sann_m=j
    sr_num=NNnum_S[i]
    r_min=DR_S[i][sann_m-1]
    r_max=DR_S[i][sr_num-1]
    for n in range(Nmax):
      r_mid=(r_min+r_max)/2
      sum_rm_min=0.0
      sum_rm_max=0.0
      sum_rm_mid=0.0
      for m in range(sann_m):
        sum_rm_min=sum_rm_min+math.acos(DR_S[i][m]/r_min)
        sum_rm_max=sum_rm_max+math.acos(DR_S[i][m]/r_max)
        sum_rm_mid=sum_rm_mid+math.acos(DR_S[i][m]/r_mid)
      f_max=sum_rm_max-np.pi
      f_min=sum_rm_min-np.pi
      f_mid=sum_rm_mid-np.pi
      if abs(f_mid) < 1.0e-10:
        break
      if f_mid*f_min > 0.0:
        r_min=r_mid
        f_min=f_mid
      else:
        r_max=r_mid
        f_max=f_mid

    r_sann_m[i] = r_mid
    if r_sann_m[i] < DR_S[i][sann_m]:
      NNnum[i]=sann_m
      for k in range(NNnum[i]):
          #NN[i][k]=NN_S[i][k]
          NN1[i][k]=NN_S[i][k]
      break
    else:
      sann_m=sann_m+1


### 4th step --- 2ND NN --- 2D SANN detection of NN by bisection method
print('2ndNN開始')
      
r_sann_m2=[0 for i in range(N)]
for i in range(N):
  for j in range(NNnum[i]+3,NNnum_S[i]):
    sann_m=j
    sr_num=NNnum_S[i]
    r_min=DR_S[i][sann_m-1]
    r_max=DR_S[i][sr_num-1]
    for n in range(Nmax):
      r_mid=(r_min+r_max)/2
      sum_rm_min=0.0
      sum_rm_max=0.0
      sum_rm_mid=0.0
      for m in range(NNnum[i],sann_m):
        sum_rm_min=sum_rm_min+math.acos(DR_S[i][m]/r_min)
        sum_rm_max=sum_rm_max+math.acos(DR_S[i][m]/r_max)
        sum_rm_mid=sum_rm_mid+math.acos(DR_S[i][m]/r_mid)
      f_max=sum_rm_max-2.0*np.pi
      f_min=sum_rm_min-2.0*np.pi
      f_mid=sum_rm_mid-2.0*np.pi
      if abs(f_mid) < 1.0e-10:
        break
      if f_mid*f_min > 0.0:
        r_min=r_mid
        f_min=f_mid
      else:
        r_max=r_mid
        f_max=f_mid

    r_sann_m2[i]=r_mid
    if r_sann_m2[i] < DR_S[i][sann_m]:
      NNnum2[i]=sann_m - NNnum[i]
 #     NNnum2[i]=sann_m
      l=NNnum[i]
      for k in range(NNnum[i],NNnum[i]+NNnum2[i]):
          #NN[i][k]=NN_S[i][k]
          NN2[i][k-l]=NN_S[i][k]
      break
    else:
      sann_m=sann_m+1

#########################################################################
### 5th step --- 3RD NN --- 2D SANN detection of NN by bisection method
##第三近接粒子を求める、第２第３合わせた粒子数がcutoffの第２近接にが対応するかどうか
print('3ndNN開始')

NNnum3=[0 for i in range(N)]
NNnum23=[0 for i in range(N)]
r_sann_m3=[0 for i in range(N)]

for i in range(N):
    for j in range(NNnum[i]+NNnum2[i]+3,NNnum_S[i]):        
        sann_m=j
        sr_num=NNnum_S[i]
        r_min=DR_S[i][sann_m-1]
        r_max=DR_S[i][sr_num-1]
        for n in range(Nmax):
            r_mid=(r_min+r_max)/2
            sum_rm_min=0.0
            sum_rm_max=0.0
            sum_rm_mid=0.0
            for m in range(NNnum[i]+NNnum2[i],sann_m):
                sum_rm_min=sum_rm_min+math.acos(DR_S[i][m]/r_min)
                sum_rm_max=sum_rm_max+math.acos(DR_S[i][m]/r_max)
                sum_rm_mid=sum_rm_mid+math.acos(DR_S[i][m]/r_mid)
            f_max=sum_rm_max-3.0*np.pi
            f_min=sum_rm_min-3.0*np.pi
            f_mid=sum_rm_mid-3.0*np.pi
            if abs(f_mid) < 1.0e-10:
                break
            if f_mid*f_min > 0.0:
                r_min=r_mid
                f_min=f_mid
            else:
                r_max=r_mid
                f_max=f_mid

        r_sann_m3[i]=r_mid
        if r_sann_m3[i] < DR_S[i][sann_m]:
            NNnum3[i]=sann_m - NNnum[i] - NNnum2[i]
            NNnum23[i]=sann_m - NNnum[i]
        #     NNnum2[i]=sann_m
            l=NNnum[i]+NNnum2[i]
            for k in range(NNnum[i]+NNnum2[i],NNnum[i]+NNnum2[i]+NNnum3[i]):
                #NN[i][k]=NN_S[i][k]
                NN3[i][k-l]=NN_S[i][k]
            break
        else:
            sann_m=sann_m+1


#########################################################################
#######rerative r_sann_m 相対半径求める

re_r_sann_m =[r_sann_m[i]/Sigma_small/2 for i in range(N)]
re_r_sann_m2=[r_sann_m2[i]/Sigma_small/2 for i in range(N)]
re_r_sann_m3=[r_sann_m3[i]/Sigma_small/2 for i in range(N)]


for i in range(N):
    if sigma_list[i] == Sigma_small:
        re_r_sann_m[i] = re_r_sann_m[i]*1.1875
        re_r_sann_m2[i] = re_r_sann_m2[i]*1.0909
        re_r_sann_m3[i] = re_r_sann_m3[i]*1.06


#########################################################################

###SANN法のカットオフ半径を出力
#f_r='output/sann_r123_'+SVC+par_occ+RorA+'.dat'
f_r='./dataset/0800/cutoff/'+file_num+'.dat'
with open(f_r, 'w') as f0:
    for i in range(N):
        f0.write(str(r_sann_m[i]/Sigma_small/2)+' '+str(r_sann_m2[i]/Sigma_small/2)+' '+str(r_sann_m3[i]/Sigma_small/2)+'\n')



###SANN法の相対カットオフ半径を出力
f_r='./dataset/0800/re_cutoff/'+file_num+'.dat'
with open(f_r, 'w') as f0:
    for i in range(N):
        f0.write(str(re_r_sann_m[i])+' '+str(re_r_sann_m2[i])+' '+str(re_r_sann_m3[i])+'\n')
##########################################################################NNnumメモる


"""
#f_NNnum='output/NNnum123_'+SVC+par_occ+RorA+'.dat'
f_NNnum='./output_cutoff_check/0780_check/NNnum123_'+'test_rcut075'+'.dat'
with open(f_NNnum, 'w') as f1:
    for i in range(N):
        f1.write(str(NNnum[i])+' '+str(NNnum2[i])+' '+str(NNnum3[i])+'\n')
"""


##########################################################################NN1メモる
#f_NN='output/NN1sann_'+SVC+par_occ+RorA+'.dat'
f_NN='./dataset/0800/NN1_list/'+file_num+'.dat'
with open(f_NN, 'w') as f2:
    for i in range(N):
        for j in range(10):
            f2.write(str(NN1[i][j])+' ')
        f2.write('\n')
##########################################################################NN1メモる
#f_NN='output/NN2sann_'+SVC+par_occ+RorA+'.dat'
f_NN='./dataset/0800/NN2_list/'+file_num+'.dat'
with open(f_NN, 'w') as f3:
    for i in range(N):
        for j in range(20):
            f3.write(str(NN2[i][j])+' ')
        f3.write('\n')
##########################################################################NN1メモる
#f_NN='output/NN3sann_'+SVC+par_occ+RorA+'.dat'
f_NN='./dataset/0800/NN3_list/'+file_num+'.dat'
with open(f_NN, 'w') as f4:
    for i in range(N):
        for j in range(32):
            f4.write(str(NN3[i][j])+' ')
        f4.write('\n')
###NN1,NN2,NN3[i]にはi番目の粒子について第1,2,3近接粒子の候補粒子が距離の近い順から格納されている
###NNnumは各粒子に対して近接粒子と判定された粒子の数であり、この数だけNNから取り出せば近接粒子の番号がわかる        


print(NN1[0][:])

"""
if par_occ == 0.72:failname = "output/NNnumN_S0720_all.dat" # nu=0.720
elif par_occ == 0.76:failname = "output/NNnumN_S0760_all.dat" # nu=0.760
elif par_occ == 0.78:failname = "output/NNnumN_S0780_all.dat" # nu=0.780
elif par_occ == 0.0:failname = "output/NNnumN_snap_3_6pi.dat" # nu=0.780
with open(failname, 'w') as f:
    for i in NNnum3:
        f.write(str(i)+'\n')
"""

print(NN3[0][:])