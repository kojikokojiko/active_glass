# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:41:28 2022

@author: 81905
"""

import numpy as np
import math
import cmath
import time
import copy
from numpy import linalg as LA
import pickle


#--------------------------------周期境界条件
def ac(x,y):
    if x > LX:
        x = x - LX
    if x < 0:
        x = x + LX
    if y > LY:
        y = y - LY
    if y < 0:
        y = y + LY 
    return(x,y)

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

def bc_2d(bc_x,bc_y):
    for i in range(0,gd_x):
        bc_x[i]=i
        bc_x[-i-1]=gd_x-i-1
        bc_x[gd_x+i]=i
    for i in range(0,gd_y):
        bc_y[i]=i
        bc_y[-i-1]=gd_y-i-1
        bc_y[gd_y+i]=i
        
#-----------------------grid_mapping

##========粒子i(Nまで)の座標をグリッド上の座標に置き換え、gd_mapに位置を記憶
##========その後、iを指定したら粒子iのグリッド上でのx座標、y座標を返すgd_mx,myに座標を保存
def gd_mp(gd_map, gd_mx, gd_my):    
    for i in range(N):
        x = table[i][0]
        y = table[i][1]
        gx = int(x*gd_x/Lx)
        gy = int(y*gd_y/Ly)
        gd_map[gx][gy] = i
        gd_mx[i] = gx
        gd_my[i] = gy
##========================

##==============
def ls5_grid(ls5_gd):
    for i in range(0,N):
        for m in range(0,60):
            j=gd_map[bc_x[gd_mx[i]+ls5_x[m]]][bc_y[gd_my[i]+ls5_y[m]]]
            if j != -1:
                ls5_gd[m][i]=j
                ls5_gd[119-m][j]=i
                

##===============引数の配置データxに含まれる全ての点について原点とのなす角を計算
def radian(x0,y0,x1,y1):
    dx01=x1-x0
    dy01=y1-y0
    dx01,dy01 = bc(dx01,dy01)###周期境界
    c = complex(dx01,dy01)###極座標で表記
    rad = cmath.phase(c)###角度求める
    return(rad)

def sort1(x):
    return(radian(0,0,x[:][0],x[:][1]))  
##======================================


##================3点がつくる角度を内積やノルムから計算し、rad→degに変換
def tangent_angle(p0,p1,p2):
    u = ([p1[0]-p0[0],p1[1]-p0[1]])
    v = ([p2[0]-p0[0],p2[1]-p0[1]])
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
##=================================プログラム内では使われない



def kouten(x1,x2,r21,r0,j):
    c=((x2[2]+r0)**2-(x1[2]+r0)**2+r21**2)*0.5/r21
    h0=math.sqrt((x2[2]+r0)**2-c**2)######に成分系の時に注意
    
    vec0=[(x1[0]-x2[0])*c/r21,(x1[1]-x2[1])*c/r21]
    vec_c=math.sqrt(vec0[0]**2+vec0[1]**2)
    #vec2=[ vec0[1]/vec_c ,-vec0[0]/vec_c]
    vec2=[ vec0[1]/c ,-vec0[0]/c]
    vec1=[-vec0[1]/vec_c , vec0[0]/vec_c]
    kou1=[x2[0]+vec0[0]+h0*vec1[0],x2[1]+vec0[1]+h0*vec1[1],j]
    kou2=[x2[0]+vec0[0]+h0*vec2[0],x2[1]+vec0[1]+h0*vec2[1],j]
    return([kou1,kou2])

def kouten1(x1,x2,sort0,r21,r0):
    c=((x2[2]+r0)**2-(x1[2]+r0)**2+r21**2)*0.5/r21
    h0=math.sqrt((x2[2]+r0)**2-c**2)######に成分系の時に注意
    
    vec0=[(x1[0]-x2[0])*c/r21,(x1[1]-x2[1])*c/r21]
    #vec_c=math.sqrt(vec0[0]**2+vec0[1]**2)
    #vec2=[ vec0[1]/vec_c ,-vec0[0]/vec_c]
    vec2=[ vec0[1]/c ,-vec0[0]/c]
    #vec1=[-vec0[1]/vec_c , vec0[0]/vec_c]
    #kou1=[x2[0]+vec0[0]+h0*vec1[0],x2[1]+vec0[1]+h0*vec1[1]]
    kou2=[x2[0]+vec0[0]+h0*vec2[0],x2[1]+vec0[1]+h0*vec2[1]]
    
    ###回転の向き的に必ず垂直なベクトルはvec2になる
    return(kou2)

##===================p2にとってp3は左側にあるか、右側にあるかを外積の正負(右ねじの法則に)で判断    
def around(p1,p2,p3):
    u = [p2[0]-p1[0],p2[1]-p1[1]]
    v = [p3[0]-p1[0],p3[1]-p1[1]]
    w = np.cross(u,v)
    return w   #左側なら正、右側なら負
##===================================

def polygon_area(sort0):
    po_area=0
    for j in range(len(sort0)):             
        po_area += sort0[j-1][0]*sort0[j][1]-sort0[j][0]*sort0[j-1][1]
    po_area = abs(po_area)*0.5
    return(po_area)
    
def triangle_area(tr1,tr2,tr3):
    tr_area=abs(tr1[0]*(tr2[1]-tr3[1])+tr2[0]*(tr3[1]-tr1[1])+tr3[0]*(tr1[1]-tr2[1]))*0.5
    return(tr_area)

def rad_vec(x0,x_st,x1):
    dx01=x1[0]-x0[0]
    dy01=x1[1]-x0[1]
    dx0st=x_st[0]-x0[0]
    dy0st=x_st[1]-x0[1]
    u = np.array([dx01, dy01])
    v = np.array([dx0st, dy0st])
    inn = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = inn / n
    a = np.arccos(np.clip(c, -1.0, 1.0))
    return(a)
    
def ougi_area(r00,r,x0,x1,x2):
    theta=rad_vec(x0,x1,x2)
    ougigata=0.5*((r00+r)**2)*theta
    return(ougigata)

def ougi_ko(r00,r,x_st,x0,x1,x2):
    theta=rad_vec(x0,x1,x2)
    ougigata_ko=(r00+r)*theta
    
    return(ougigata_ko)
    
SVC='S'
pi = math.pi
n = 4096
for i in range(10000000000):
    if i**2 < n:
        pass
    else:
        N_all = i
        break

N = 4096
for i in range(10000000000):
    if i**2 < n:
        pass
    else:
        N_all = i
        break
n1 = 2731
n2 = 1365
x1 = n1/n
x2 = n2/n
a = 1.4
pi = np.pi
par_occ='0800'
pack = float(par_occ)/1000
par_occ1 = str(pack)
Lx = 64.0
Ly = Lx
LX = Lx
LY = Ly
#sigma = np.loadtxt('sigma_list.dat')
Mol_small, Mol_large = 2, 1 ###モル比
Sigma_ratio = 1.4
Sigma_small = np.sqrt(pack*64*64*(Mol_large+Mol_small)/4096/np.pi/(Mol_large*Sigma_ratio**2 + Mol_small))
Sigma_large = Sigma_ratio * Sigma_small
sigma = np.full(N, Sigma_large)
sigma_decision = []
with open('sigma_init' + '.txt') as f:
    for line in f:
        sigma_decision_element = float(line)
        sigma_decision.append(sigma_decision_element)
for i in range(N):
    if sigma_decision[i] < 0.1:
        sigma[i] = Sigma_small
#sigma = sigma.reshape((4096,1))###このプログラムではこの形にせずに1次元で扱うべき



#小さいsigma
#sigma1 = sigma[0]
sigma1 = Sigma_small
#大きいsigma
#sigma2 = sigma[1]
sigma2 = Sigma_large

frame0 = 1
skip=N*frame0

config_number = str(1)
file_type = 'inh'









#print(table11[0][1225])
#table = pos[5]
first=time.time()
        
NNmax=80
gd_x = int(Lx / (math.sqrt(2)*sigma1) + 1)
gd_y = int(Ly / (math.sqrt(2)*sigma1) + 1)
gd_mx = [0 for i in range(0, N)]
gd_my = [0 for i in range(0, N)]
bc_x=[0 for i in range(-gd_x,gd_x+gd_x)]
bc_y=[0 for i in range(-gd_y,gd_y+gd_y)]
bc_2d(bc_x,bc_y)
ls5_x=[-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,
       -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,
       -3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,
       -2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,
       -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        0, 0, 0, 0, 0,    0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

ls5_y=[-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1,    1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,
       -5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5]

SANN_rcut = sigma1 *2
SANN_rcut2 = sigma2 *2

#SANN_rcut = 0.7*2*3
#SANN_rcut2 = 0.7*2*3

    ###### For efficiency
NN_all = [0 for i in range(frame0)]

angle_all = [0 for i in range(frame0)]

#for dis in range(1,101):

table1 = [[] for i in range(frame0)]
pos_all = np.loadtxt('./dataset/'+par_occ+'/initial_pos/'+file_type+'/'+config_number+'.dat')

for i in range(frame0):
    for k in range(4096):
        table1[i].append(pos_all[i*4096 + k])
table11 =[[0 for i in range(N)]for k in range(frame0)]
for t in range(frame0):
    for i in range(N):
        table11[t][i] = [table1[t][i][0],table1[t][i][1]]
        table11[t][i].append(sigma[i])
        
area_all = [0 for t in range(frame0)]
NN_all = [0 for t in range(frame0)]
angle_all = [0 for t in range(frame0)]
sort_all = [0 for t in range(frame0)]
NN_table_all = [0 for t in range(frame0)]
        
##### MAIN ROUTINE (SANN) #####
for frame in range(frame0):
    print('frame',frame)
    first=time.time()
    table = table11[frame]
    
    
    gd_map = [[-1 for i in range(0, gd_y)] for j in range(0, gd_x)] 
    gd_mp(gd_map,gd_mx,gd_my)
    ls5_gd=[[-1 for i in range(N)] for k in range(121)]
    ls5_grid(ls5_gd)

    NNnum=[0 for i in range(N)]
    NNnum_S=[0 for i in range(N)]
    DR_S=[[0 for i in range(NNmax)] for j in range(N)]
    NN_S=[[-1 for i in range(NNmax)] for j in range(N)]
    X_Y=[[0 for i in range(NNmax)] for j in range(N)]
    NNnum2=[0 for i in range(N)]
    NNnum_S2=[0 for i in range(N)]
    DR_S2=[[0 for i in range(NNmax)] for j in range(N)]
    NN_S2=[[-1 for i in range(NNmax)] for j in range(N)]
    X_Y2=[[0 for i in range(NNmax)] for j in range(N)]
    
    
    
    
##=====grid上で各粒子iの近傍の粒子を大まかなカットオフ半径にかけて1次と2次までの近傍を近い順も関係なく特定
##=====NNnum_S(近傍粒子の数)の値を+1ずつしてループ繰り返すことで粒子iの近傍に関する情報を追加していく
##=====rcut2のif部分はrcut1に含まれる部分も共通している    
    for i in range(N):
        xi=table[i][0]
        yi=table[i][1]
        for m in range(60):
            j=ls5_gd[m][i]
            if j != -1:
                xj=table[j][0]
                yj=table[j][1]
                dxij=xj-xi
                dyij=yj-yi
                if dxij > Lx/2:
                    dxij  = dxij - Lx
                elif dxij < -Lx/2:
                    dxij = dxij + Lx
                if dyij > Ly/2:
                    dyij = dyij - Ly
                elif dyij < -Ly/2:
                    dyij = dyij + Ly
                dr=np.sqrt(dxij**2+dyij**2)
                dl = dr - table[i][2] - table[j][2]
                if dl < SANN_rcut:  #2成分系注意   
                    NN_S[i][NNnum_S[i]]=j
                    NN_S[j][NNnum_S[j]]=i
                    X_Y[i][NNnum_S[i]] = [dxij,dyij]
                    X_Y[j][NNnum_S[j]] = [-dxij,-dyij]
                    DR_S[i][NNnum_S[i]]=dr-table[j][2]
                    DR_S[j][NNnum_S[j]]=dr-table[i][2]
                    NNnum_S[i]+=1
                    NNnum_S[j]+=1
                if dl < SANN_rcut2:  #2成分系注意   
                    NN_S2[i][NNnum_S2[i]]=j
                    NN_S2[j][NNnum_S2[j]]=i
                    X_Y2[i][NNnum_S2[i]] = [dxij,dyij]
                    X_Y2[j][NNnum_S2[j]] = [-dxij,-dyij]
                    DR_S2[i][NNnum_S2[i]]=dr-table[j][2]
                    DR_S2[j][NNnum_S2[j]]=dr-table[i][2]
                    NNnum_S2[i]+=1
                    NNnum_S2[j]+=1
##===========================================================
#    print(DR_S[1])
                    
 #   print(NNnum_sann1[0],NN_sann1[0])
     
    
     #   print(NNnum_S2[2154])
   
    for i in range(N):
        if NNnum_S[i] ==0 or NNnum_S2[i] == 0:
            print(i)
        NNs_list=[[DR_S[i][k],NN_S[i][k],X_Y[i][k]] for k in range(NNnum_S[i])]
        NNs_list2=[[DR_S2[i][k],NN_S2[i][k],X_Y2[i][k]] for k in range(NNnum_S2[i])]
        NNs_list.sort(key=lambda x:x[0])
        NNs_list2.sort(key=lambda x:x[0])
        #0番目の要素をキーとして小さい順にソート
        #例えば[[2,0,1],[0,1,2],[1,2,0]]→[[0,1,2],[1,2,0],[2,0,1]]という風に
        #いわゆる近い順に並び替え
#        if i == 0:
#            print(NNs_list)

        #NNs_listに基づいて元のDR_SやX_Yを注目粒子に近い順のデータに更新       
        for k in range(NNnum_S[i]):
            DR_S[i][k]=NNs_list[k][0]
            NN_S[i][k]=NNs_list[k][1]
            X_Y[i][k] = NNs_list[k][2]
        for k in range(NNnum_S2[i]):
            DR_S2[i][k]=NNs_list2[k][0]
            NN_S2[i][k]=NNs_list2[k][1]
            X_Y2[i][k] = NNs_list2[k][2]
        
  
                  
    cs1 = [[0 for j in range(NNnum_S[i]*2)]for i in range(N)]
    cs2 = [[0 for j in range(NNnum_S2[i]*2)]for i in range(N)]
    for i in range(N):
        x0 = table[i][0]
        y0 = table[i][1]
        r0 = sigma[i]
        for k in range(NNnum_S[i]):
            j = NN_S[i][k]
       #     a,b = bc(table[j][0]-x0,table[j][1]-y0)
            a,b = X_Y[i][k][0],X_Y[i][k][1]
            h = [a,b,sigma[j]]
            r = DR_S[i][k]+table[j][2]
            z2 = kouten([0,0,r0],h,r,sigma1,j)
          #  for h in range(len(z2)):
           #     z2[h][0],z2[h][1] = ac(z2[h][0]+x0,z2[h][1]+y0)
         #   cs2[i][k] = [z2,j]
            cs1[i][k*2] = z2[0]
            cs1[i][k*2+1] = z2[1]
        cs1[i].sort(key=sort1)
        #交点2つ(x,y,相手の粒子番号j)ずつ求めてcs1に格納、最後にsort１のように並び替え
        
        for k in range(NNnum_S2[i]):
            j = NN_S2[i][k]
       #     a,b = bc(table[j][0]-x0,table[j][1]-y0)
            a,b = X_Y2[i][k][0],X_Y2[i][k][1]
            h = [a,b,sigma[j]]
            r = DR_S2[i][k]+table[j][2]
            z2 = kouten([0,0,r0],h,r,sigma2,j)
          #  for h in range(len(z2)):
           #     z2[h][0],z2[h][1] = ac(z2[h][0]+x0,z2[h][1]+y0)
         #   cs2[i][k] = [z2,j]
            cs2[i][k*2] = z2[0]
            cs2[i][k*2+1] = z2[1]
        cs2[i].sort(key=sort1)
        
        #上までの計算は実際の座標を踏まえた上でやっておらず相対的な座標から行われる
        #そのため下で注目粒子iの座標を足して(境界条件に注意)実際の座標に戻す
        for k in range(NNnum_S[i]*2):
            a,b = ac(cs1[i][k][0]+x0,cs1[i][k][1]+y0)
            cs1[i][k] = [a,b,cs1[i][k][2]]
        for k in range(NNnum_S2[i]*2):
            a,b = ac(cs2[i][k][0]+x0,cs2[i][k][1]+y0)
            cs2[i][k] = [a,b,cs2[i][k][2]]
#    print(cs1[0])
   
    
    NN_new = [0 for i in range(N)]
    NN_table = [0 for i in range(N)]
    area=[0 for i in range(N)]
    fs=[0 for i in range(N)] 
    angle = [0 for i in range(N)]
    area_2=[0 for i in range(N)]
    sor = [0 for i in range(N)]
    
    fs_normal = [0 for i in range(N)]
    area_normal = [0 for i in range(N)]
    area_h = [0 for i in range(N)]
            
    ### 確定粒子検出　###
    for i in range(N):
       sort_cs = [0] 
       NN = []  
       NN_fig = []
       in_array = []
       x0 = table[i][0]
       y0 = table[i][1]
       r0 = table[i][2]
       NG_num = i
      
       if r0 == sigma1:
          cs = cs1
       elif r0 == sigma2:
           cs = cs2
       c1 = NN_S[i][0] #一つ目の確定粒子
      
       
       a1,b1 = bc(table[c1][0]-x0,table[c1][1]-y0)
  #     NN.append([a1,b1,table[c1][2],c1])
       R = [0 for i in range(len(cs[c1]))]
       for k in range(len(cs[c1])):
             a,b = bc(cs[c1][k][0]-x0,cs[c1][k][1]-y0)
             c,d = bc(cs[c1][k][0]-table[c1][0],cs[c1][k][1]-table[c1][1])
             r = math.sqrt(a**2+b**2)
             R[k] = [r,k]
       R.sort(key=lambda x:x[0])
       if cs[c1][R[0][1]][2] != NG_num:
          c2 = cs[c1][R[0][1]][2] #二つ目の確定粒子
          c2_num = R[0][1]
       elif cs[c1][R[1][1]][2] != NG_num:
           c2 = cs[c1][R[1][1]][2]
           c2_num = R[1][1]
       else:
           if len(cs[c1]) <3:
               print(i)
           c2 = cs[c1][R[2][1]][2]
           c2_num = R[2][1]
       a2,b2 = bc(table[c2][0]-x0,table[c2][1]-y0)
      # NN.append([a2,b2,table[c2][2],c2])
       #NN.sort(key=sort1)
       a,b = bc(cs[c1][c2_num][0]-x0,cs[c1][c2_num][1]-y0)
      
       if around([a,b],[a1,b1],[a2,b2]) > 0: #c2がc1の左に存在
          NN_fig = [c1,c2]
       else:
          NN_fig = [c2,c1] 
       a1,b1 = bc(table[NN_fig[0]][0]-x0,table[NN_fig[0]][1]-y0)
       a2,b2 = bc(table[NN_fig[1]][0]-x0,table[NN_fig[1]][1]-y0)
       NN.append([a1,b1,table[NN_fig[0]][2],NN_fig[0]])
       NN.append([a2,b2,table[NN_fig[1]][2],NN_fig[1]])
       sort_cs[0] = [a,b,NN_fig[0],NN_fig[1]]
      
      
       
     
      
      
       abc = 0
       start = NN_fig[0]
       while abc == 0:
             k = len(sort_cs)-1
             ne = sort_cs[k][3]
             l = []
             for k in range(len(cs[ne])):
                 if cs[ne][k][2] == start:
                    a,b = bc(cs[ne][k][0]-x0,cs[ne][k][1]-y0)
                    dr1 = math.sqrt((a-sort_cs[-1][0])**2+(b-sort_cs[-1][1])**2)
                    dr = math.sqrt(a**2+b**2)
                    l.append([cs[ne][k],dr1,k])
             l.sort(key=lambda x:x[1])
        #  f = cs1[ne].index([sort_cs[k][0],sort_cs[k][1],start])
            
             f = l[0][2]
             if cs[ne][f-1][2] != NG_num:
                next_c = cs[ne][f-1][2]
                next_cs = cs[ne][f-1]
             elif cs[ne][f-2][2] != NG_num:
                 next_c = cs[ne][f-2][2]
                 next_cs = cs[ne][f-2]    
             else:
                 next_c = cs[ne][f-3][2]
                 next_cs = cs[ne][f-3]
           
             
             if ne == sort_cs[0][2] and next_c == sort_cs[0][3]:
                abc = 1
                break
   # print(next_c in NN_fig)
             if next_c not in NN_fig: 
                NN_fig.append(next_c)
             c,d = bc(table[next_c][0]-x0,table[next_c][1]-y0)
             NN.append([c,d,table[next_c][2],next_c])
             a,b = bc(next_cs[0]-x0,next_cs[1]-y0)
             sort_cs.append([a,b,ne,next_c])
             
             start = ne
      # NN_fig = list(set(NN_fig))
     #  NN.sort(key=sort1)
   
       NN.pop(-1)
       NN_new[i] = NN_fig
       NN_table[i] = NN
       NN = NN_table[i]
       sor[i] = sort_cs          
       pol = []
       for k in range(len(NN)):
           pol.append(NN[k])
           pol.append(sort_cs[k])
       s = polygon_area(pol)
       s1 = 0
       for k in range(len(NN)):
           s1 += ougi_area(NN[k][2],r0,NN[k],sort_cs[k-1],sort_cs[k])
       s2 = s-s1
       area_2[i] = s2
       
       for k in range(len(NN)):
           r21 = math.sqrt((NN[k-1][0]-NN[k][0])**2+(NN[k-1][1]-NN[k][1])**2)
         #  if r21 > sigma[i]*2 + sigma[NN[k-1][3]] + sigma[NN[k][3]]:
            #   print(NN_fig)
             #  print(i,NN[k-1][3],NN[k][3],r21,NN[k-1],NN[k])
           ins=kouten1(NN[k-1],NN[k],NN,r21,r0)
           in_array.append(copy.deepcopy(ins))
       poly_area=polygon_area(NN) 
       area_1=0   
       
       for k in range(len(NN)):
            area_1 += triangle_area(NN[k-1],NN[k],in_array[k])
            area_1 += ougi_area(NN[k-1][2],r0,NN[k-1],in_array[k-1],in_array[k])
        #    print(poly_area - area_1)
            area[i] = poly_area - area_1
            fs[i]+=ougi_ko(r0,NN[k-1][2],[0,0],NN[k-1][0:2],in_array[k-1][:],in_array[k][:])
       if area_2[i] < 0:
           print(i,NN_fig,area[i],area_2[i])
       
       angle0 = [0 for i in range(len(NN))]
       for j in range(len(NN)):
            if j==0:
                theta1_0=radian(NN[j-1][0],NN[j-1][1],in_array[j][0],in_array[j][1])
                theta2_0=radian(NN[j-1][0],NN[j-1][1],in_array[j-1][0],in_array[j-1][1])
                angle0[j-1]=[theta1_0,theta2_0]
            else:
                theta1=radian(NN[j-1][0],NN[j-1][1],in_array[j][0],in_array[j][1])
                theta2=radian(NN[j-1][0],NN[j-1][1],in_array[j-1][0],in_array[j-1][1])
                angle0[j-1] = [theta1,theta2]
       angle[i] = angle0
       
       
    p1=[0 for j in range(N)]
    v=float(par_occ)/1000
    k = 0
    for i in range(N):
        
        
        r = sigma[i]
        fs_normal[i] = fs[i]/(2*sigma1)
        area_normal[i] = area[i]/(4*sigma1**2)#自由体積←小粒子の直径で無次元化
        sfvf=fs[i]/area[i]
        z=1+(r)*sfvf/2
        p1[i]=z*4*v/math.pi
        area_h[i] = area_normal[i]/(math.pi*sigma[i]**2)#相対自由体積
        
#        print(angle)
#        print(NN_new)

    np.save('./dataset/'+par_occ+'/fv_NN/'+file_type+'/'+config_number, NN_new)
    np.save('./dataset/'+par_occ+'/angle/'+file_type+'/'+config_number, angle)
    


    f_r='./dataset/'+par_occ+'/free_sur/'+file_type+'/'+config_number+'.dat'
    with open(f_r, 'w') as f0:
        for i in range(N):
            f0.write(str(fs_normal[i])+' ')
    f_r='./dataset/'+par_occ+'/free_vol/'+file_type+'/'+config_number+'.dat'
    with open(f_r, 'w') as f0:
        for i in range(N):
            f0.write(str(area_normal[i])+' ')
    f_r='./dataset/'+par_occ+'/re_free_vol/'+file_type+'/'+config_number+'.dat'
    with open(f_r, 'w') as f0:
        for i in range(N):
            f0.write(str(area_h[i])+' ')
    f_r='./dataset/'+par_occ+'/local_pressure/'+file_type+'/'+config_number+'.dat'
    with open(f_r, 'w') as f0:
        for i in range(N):
            f0.write(str(p1[i])+' ')

        

"""        
#            p1[i] = r * sfvf/2
          #  k += p1[i]
        j = max(p1)
      #  print(p1.index(j))
       # print(max(p1),fs[1896],area[1896],p1[1896],sigma[1896])
        #print(k)
     #   p2 = k/N
     #   print('frame',frame)
     #   print(p2)
        
        
   
        
            
        
        #print(area_ave[0])
        
        NN_all[frame] = NN_new
        area_all[frame] = area
        angle_all[frame] = angle
        NN_table_all[frame] = NN_table
        sort_all[frame] = sor
     #   area_re_all[frame] = area_h
      #  print(area[1896],area[1896])
       # print(area_2[1896],area_2[1896])
        
         
        fin=time.time()-first
        print(fin)
        
#    np.save('press_ave_'+par_occ+p,p1)
    print('over')
"""    


"""   
    with open('NN_'+par_occ+'_dis'+p+'.txt', 'wb') as p_data072:
        pickle.dump(NN_all, p_data072)
    with open('angle_'+par_occ+'_dis'+p+'.txt', 'wb') as p_data072:
        pickle.dump(angle_all, p_data072)
    with open('NNtable_'+par_occ+'_dis'+p+'.txt', 'wb') as p_data072:
        pickle.dump(NN_table_all, p_data072)
    with open('sort_'+par_occ+'_dis'+p+'.txt', 'wb') as p_data072:
        pickle.dump(sort_all, p_data072)
    with open('area_'+par_occ+'_dis'+p+'.txt', 'wb') as p_data072:
         pickle.dump(area_all, p_data072)
"""
    
