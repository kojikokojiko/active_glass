# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:49:48 2021

@author: 81905
"""

import os, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from numba import jit
from copy import deepcopy
from PIL import Image





"""関数定義"""





###全衝突時間計算
@jit
def compute_next_event_all(pos, vel, event_list, eepgm_grid, eepgm_list, event_list_arg, sigma_list):
    event_list[:,:] = 9999999.0
    event_list_arg[:,:] = N
    for i in range(N):
        mask_n = 0
        for j, k in mask:
            ngx = j + eepgm_list[i][0]
            ngy = k + eepgm_list[i][1]
            d_x = 0.0
            d_y = 0.0
         
            if ngx < 0:
                ngx += N_gx
                d_x = 1.0
            elif ngx >= N_gx:
                ngx -=N_gx
                d_x = -1.0
            if ngy < 0:
                ngy += N_gy
                d_y = 1.0
            elif ngy >= N_gy:
                ngy -=N_gy
                d_y = -1.0
        
            if not eepgm_grid[ngx][ngy] == N:
                del_x = pos[i] - pos[eepgm_grid[ngx][ngy]] 
                del_x[0] += L_x * d_x
                del_x[1] += L_y * d_y
                del_v = vel[i]-vel[eepgm_grid[ngx][ngy]]
                scal = np.dot(del_x, del_v)
                upsilon = scal ** 2 - np.dot(del_v, del_v) * (np.dot(del_x, del_x) -  (sigma_list[i] + sigma_list[eepgm_grid[ngx][ngy]]) ** 2)
                if upsilon > 0.0 and scal < 0.0:
                    colision_time = - (scal + np.sqrt(upsilon)) / np.dot(del_v, del_v)
                    event_list[i][mask_n] = colision_time
                event_list_arg[i][mask_n] = eepgm_grid[ngx][ngy]
                


                
            mask_n += 1

###衝突時間計算
@jit            
def compute_next_event(pos, vel, event_list, eepgm_grid, eepgm_list, colison_list, next_event_arg, sigma_list, next_event):
    coliN = 0
    colison_list[:] = N
    for i in next_event_arg:
        
        mask_n = 0
        for j, k in mask:
            ngx = j + eepgm_list[i][0]
            ngy = k + eepgm_list[i][1]
            if ngx < 0:
                ngx += N_gx
            elif ngx >= N_gx:
                ngx -=N_gx
            if ngy < 0:
                ngy += N_gy
            elif ngy >= N_gy:
                ngy -=N_gy
                
            if not eepgm_grid[ngx][ngy] == N:
                colison_list[mask_n + coliN * N_mask] = eepgm_grid[ngx][ngy]
                del_x = pos[i] - pos[eepgm_grid[ngx][ngy]]
                if del_x[0] > L_x / 2:
                    del_x[0] -=L_x
                elif del_x[0] < -L_x / 2:
                    del_x[0] +=L_x
                if del_x[1] > L_y / 2:
                    del_x[1] -=L_y
                elif del_x[1] < -L_y / 2:
                    del_x[1] +=L_y
                del_v = vel[i]-vel[eepgm_grid[ngx][ngy]]
                scal = np.dot(del_x, del_v)
                upsilon = scal ** 2 - np.dot(del_v, del_v) * (np.dot(del_x, del_x) - (sigma_list[i] + sigma_list[eepgm_grid[ngx][ngy]]) ** 2)
                
                if upsilon > 0.0 and scal < 0.0:
                    coli_time = - (scal + np.sqrt(upsilon)) / np.dot(del_v, del_v) 
#                    event_list[eepgm_grid[ngx][ngy]][N_mask-mask_n-1] = event_list[i][mask_n] = coli_time
                    if coli_time > next_event:
                        event_list[eepgm_grid[ngx][ngy]][N_mask-mask_n-1] = event_list[i][mask_n] = coli_time
                    else:
                        event_list[eepgm_grid[ngx][ngy]][N_mask-mask_n-1] = event_list[i][mask_n] = 9999999.0
                        
                else:
                    event_list[eepgm_grid[ngx][ngy]][N_mask-mask_n-1] = event_list[i][mask_n] = 9999999.0

            mask_n += 1
        coliN += 1

###single_event_list CBTの更新
@jit
def compute_CBT(colison_list, single_event_list, single_event_list_arg, event_list, event_list_arg, CBT_list, next_event_arg):
    for i in colison_list:
        if not i == N:
            single_event_list[i] = np.min(event_list[i])
            single_event_list_arg[i] = event_list_arg[i][np.argmin(event_list[i])]
                
    for i in next_event_arg:
        single_event_list[i] = np.min(event_list[i])
        single_event_list_arg[i] = event_list_arg[i][np.argmin(event_list[i])]
    
    for i in colison_list:
        if not i == N:
            next_N =int((N_cbt+i-2)/2)
            for j in range(N_log):
                if single_event_list[CBT_list[2*next_N+1]] <= single_event_list[CBT_list[2*next_N+2]]:
                    CBT_list[next_N] =  CBT_list[2*next_N+1]
                else:
                    CBT_list[next_N] =  CBT_list[2*next_N+2]
                next_N = int((next_N-1)/2)
    
    for i in next_event_arg:
        next_N =int((N_cbt+i-2)/2)
        for j in range(N_log):
            if single_event_list[CBT_list[2*next_N+1]] <= single_event_list[CBT_list[2*next_N+2]]:
                CBT_list[next_N] =  CBT_list[2*next_N+1]
            else:
                CBT_list[next_N] =  CBT_list[2*next_N+2]
            next_N = int((next_N-1)/2)        

    
###CBT作成
@jit
def compute_CBT_all(single_event_time, CBT_list):
    for i in range(N_cbt - 2, -1 ,-1):
        if single_event_time[CBT_list[2*i+1]] <= single_event_time[CBT_list[2*i+2]]:
            CBT_list[i] =  CBT_list[2*i+1]
        else:
            CBT_list[i] =  CBT_list[2*i+2]


###衝突粒子の位置速度更新
@jit
def compute_colision_pair(del_t, pos, vel, next_event_arg, pos_calculate):
    for i in next_event_arg:
        pos[i] += vel[i] * del_t
        pos_calculate[i] += vel[i] * del_t
        if pos[i][0] <=0:
            pos[i][0] += L_x
        elif pos[i][0]> L_x:
            pos[i][0] -=L_x
        if pos[i][1] <=0:
            pos[i][1] += L_y
        elif pos[i][1]> L_y:
            pos[i][1] -=L_y        
            
    a, b = next_event_arg 
    del_x = pos[b]- pos[a]
    if del_x[0] > L_x / 2:
        del_x[0] -=L_x
    elif del_x[0] < -L_x / 2:
        del_x[0] +=L_x
    if del_x[1] > L_y / 2:
        del_x[1] -=L_y
    elif del_x[1] < -L_y / 2:
        del_x[1] +=L_y
    abs_x = np.linalg.norm(del_x)
    e_perp =  del_x / abs_x
    del_v = vel[b] - vel[a]
    scal = np.dot(del_v, e_perp)
    vel[a] += (1.0+Elast) / 2 * e_perp * scal
    vel[b] -= (1.0+Elast) / 2 * e_perp * scal
    
    for i in next_event_arg:
        pos[i] -= vel[i] *del_t
        pos_calculate[i] -= vel[i] * del_t
        if pos[i][0] <=0:
            pos[i][0] += L_x
        elif pos[i][0]> L_x:
            pos[i][0] -=L_x
        if pos[i][1] <=0:
            pos[i][1] += L_y
        elif pos[i][1]> L_y:
            pos[i][1] -=L_y        


###位置更新
@jit(cache = True)
def compute_pos_all(del_t, pos, vel, pos_calculate, eepgm_grid):  
    pos += vel * del_t
    pos_calculate += vel * del_t
    for i in range(N_gx):
        for j in range(int(R_mask/2)):
            if not eepgm_grid[i][j] == N:
                if pos[eepgm_grid[i][j]][0] <=0:
                    pos[eepgm_grid[i][j]][0] += L_x
                elif pos[eepgm_grid[i][j]][0]> L_x:
                    pos[eepgm_grid[i][j]][0] -=L_x
                if pos[eepgm_grid[i][j]][1] <=0:
                    pos[eepgm_grid[i][j]][1] += L_y
                elif pos[eepgm_grid[i][j]][1]> L_y:
                    pos[eepgm_grid[i][j]][1] -=L_y    
            if not eepgm_grid[i][N_gy-j-1] == N:
                if pos[eepgm_grid[i][N_gy-j-1]][0] <=0:
                    pos[eepgm_grid[i][N_gy-j-1]][0] += L_x
                elif pos[eepgm_grid[i][N_gy-j-1]][0]> L_x:
                    pos[eepgm_grid[i][N_gy-j-1]][0] -=L_x
                if pos[eepgm_grid[i][N_gy-j-1]][1] <=0:
                    pos[eepgm_grid[i][N_gy-j-1]][1] += L_y
                elif pos[eepgm_grid[i][N_gy-j-1]][1]> L_y:
                    pos[eepgm_grid[i][N_gy-j-1]][1] -=L_y               
                
    for i in range(N_gy):
        for j in range(int(R_mask/2)):
            if not eepgm_grid[j][i] == N:
                if pos[eepgm_grid[j][i]][0] <=0:
                    pos[eepgm_grid[j][i]][0] += L_x
                elif pos[eepgm_grid[j][i]][0]> L_x:
                    pos[eepgm_grid[j][i]][0] -=L_x
                if pos[eepgm_grid[j][i]][1] <=0:
                    pos[eepgm_grid[j][i]][1] += L_y
                elif pos[eepgm_grid[j][i]][1]> L_y:
                    pos[eepgm_grid[j][i]][1] -=L_y    
            if not eepgm_grid[N_gx-j-1][i] == N:
                if pos[eepgm_grid[N_gx-j-1][i]][0] <=0:
                    pos[eepgm_grid[N_gx-j-1][i]][0] += L_x
                elif pos[eepgm_grid[N_gx-j-1][i]][0]> L_x:
                    pos[eepgm_grid[N_gx-j-1][i]][0] -=L_x
                if pos[eepgm_grid[N_gx-j-1][i]][1] <=0:
                    pos[eepgm_grid[N_gx-j-1][i]][1] += L_y
                elif pos[eepgm_grid[N_gx-j-1][i]][1]> L_y:
                    pos[eepgm_grid[N_gx-j-1][i]][1] -=L_y     



###グリッド更新
@jit
def compute_eepgm(eepgm_grid, eepgm_list, pos):
    eepgm_grid[:,:] = N
    for i in range(N):
        eepgm_grid[np.int(pos[i][0]/L_gx)][np.int(pos[i][1]/L_gy)]=i
        eepgm_list[i][0] = np.int(pos[i][0]/L_gx)
        eepgm_list[i][1] = np.int(pos[i][1]/L_gy)


def vel_reset(vel):
    vel_x=0
    vel_y=0
    for i in range(N):
        vel_x+=vel[i][0]
        vel_y+=vel[i][1]
    vel_x=vel_x/N  
    vel_y=vel_y/N 
    for i in range(N):
        vel[i][0]-=vel_x
        vel[i][1]-=vel_y


###スナップショット    
def snapshot(t, pos, sigma_list):
    global img

    plt.clf()
    plt.figure(figsize=(16, 16), dpi=80)
    ax = plt.axes()
    plt.xlim(0,L_x)
    plt.ylim(0,L_y)
    for i in range(N):
        x, y = pos[i]
        Sigma = sigma_list[i]
        if Sigma == Sigma_small:
            color = 'g'
        else:
            color = 'r'
        circle = patches.Circle((x, y), radius=Sigma, fc = color)
        ax.add_patch(circle)

        if x >= L_x - Sigma and Sigma < y < L_y - Sigma:
            circle = patches.Circle((x-L_x, y), radius=Sigma, fc = color)
            ax.add_patch(circle)
        elif x <= Sigma and Sigma < y < L_y - Sigma:
            circle = patches.Circle((x+L_x, y), radius=Sigma, fc = color)
            ax.add_patch(circle)
        elif y >= L_y - Sigma and Sigma < x < L_x - Sigma:
            circle = patches.Circle((x, y-L_y), radius=Sigma, fc = color)
            ax.add_patch(circle)
        elif y <= Sigma and Sigma < x < L_x - Sigma:
            circle = patches.Circle((x, y+L_y), radius=Sigma, fc = color)
            ax.add_patch(circle)
        elif x >= L_x - Sigma:
            if y >= L_y - Sigma:
                circle = patches.Circle((x-L_x, y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x, y-L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x-L_x, y-L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
            elif y <= Sigma:
                circle = patches.Circle((x-L_x, y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x, y+L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x-L_x, y+L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
        elif x <= Sigma:
            if y >= L_y - Sigma:
                circle = patches.Circle((x+L_x, y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x, y-L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x+L_x, y-L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
            elif y <= Sigma:
                circle = patches.Circle((x+L_x, y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x, y+L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
                circle = patches.Circle((x+L_x, y+L_y), radius=Sigma, fc = color)
                ax.add_patch(circle)
        else:
            pass      

    ax.set_aspect('equal')
    plt.title("t = "+f'{t:.2f}')
#    plt.savefig(snapshotout +'/'+str(img)+'.png')
    plt.savefig('gnn_test'+'.png')
    plt.close()
    img += 1
    
#def calculate_msd_and_ngp(pos_1, pos_0, outfile_msd, outfile_ngp):    
def calculate_msd_and_ngp_and_f(pos_1, pos_0):
    m2 = 0 
    m4 = 0
    for i in range(N):
        m2_element = ((pos_1[i][0] - pos_0[i][0])) ** 2 + ((pos_1[i][1] - pos_0[i][1])) ** 2
        m4_element = m2_element ** 2
        m2 += m2_element
        m4 += m4_element
    m2 = m2 / (N)
    m4 = m4 / (N)
    

    

    
    msd = m2 / ((2*Sigma_small) ** 2)
    ngp = ((m4/(m2**2))/2) - 1
    
    
    
    k = 2 * (np.pi)
    isf = np.exp(-k**2 * msd / 4)
    
    
    with open('msd__7.txt', mode="a") as f:
        f.write(str(msd)+"\n")
    with open('ngp__7.txt', mode="a") as f:
        f.write(str(ngp)+"\n")
    with open('isf__7.txt', mode="a") as f:
        f.write(str(isf)+"\n")



"""パラメータの定義"""
np.random.seed(0)

L_x = 64
L_y = 64

N_side = 64     ###1辺あたりの粒子数
N = N_side**2     ###全粒子数
Mol_small, Mol_large = 2, 1 ###モル比
Sigma_ratio = 1.4
Density = 0.79     ###占有率
senyuritu = '079'  ###出力
Elast = 1.0     ###反発係数
Sigma_small = np.sqrt(Density*L_x*L_y*(Mol_large+Mol_small)/N/np.pi/(Mol_large*Sigma_ratio**2 + Mol_small))   ###粒子（小）半径
Sigma_large = Sigma_ratio * Sigma_small
R_mask = 5    ###マスクの1辺のグリッド数

dt = 0.001    # dt=0 衝突ごとに出力
n_steps = 1 ###計算ステップ

"""出力先"""
"""
output_dir = "./study_b4/niseibun"
msdout = output_dir + "/msd_txt" + "/msd_" + senyuritu + ".txt"
ngpout = output_dir + "/ngp_txt" + "/ngp_" + senyuritu + ".txt"
snapshotout = output_dir + '/snapshot/' + senyuritu
if not os.path.exists(snapshotout):
    os.makedirs(snapshotout)
"""



###初期位置
#pos_1 = np.array([[(i+0.4)/N_side*L_x, (j+0.4)/N_side*2*L_y] for i in range(N_side) for j in range(int(N_side/2))])
#pos_2 = np.array([[(i+0.9)/N_side*L_x, (j+0.9)/N_side*2*L_y] for i in range(N_side) for j in range(int(N_side/2))])
#pos = np.concatenate([pos_1  ,pos_2])

#pos = np.array([[L_x/N_side*(1/2+i), L_y/N_side*(1/2+j)] for i in range(N_side) for j in range(N_side)])

###初期位置###


pos_element_list = np.loadtxt('1.dat')
pos = np.full((N,2), 0, dtype=float)

x_pos = [] 
#with open('x_init_mix' + '.txt') as f:
#    for line in f:
#        x_pos_element = float(line)
#        x_pos.append(x_pos_element)
        

        
y_pos = []
#with open('y_init_mix' + '.txt') as f:
#    for line in f:
#        y_pos_element = float(line)
#        y_pos.append(y_pos_element)
        
#for i in range(N):
#    pos[i][0] = x_pos[i]
#    pos[i][1] = y_pos[i]
    

for i in range(N):
    pos[i][0] = pos_element_list[i][0]
    pos[i][1] = pos_element_list[i][1]
    


###半径リスト###
sigma_list = np.full(N, Sigma_large)
sigma_decision = []
with open('sigma_init' + '.txt') as f:
    for line in f:
        sigma_decision_element = float(line)
        sigma_decision.append(sigma_decision_element)
for i in range(N):
    if sigma_decision[i] < 0.1:
        sigma_list[i] = Sigma_small    




###msd_and_ngp
#pos_0 = deepcopy(pos)
pos_msd_and_ngp = deepcopy(pos)


###初期速度
vel = np.array(np.random.randn(N,2))
vel_reset(vel)
V_max = np.max(vel)   ###最大速度

###EEPGM
N_gx = int(L_x/np.sqrt(2)/Sigma_small)+1   ###1辺のグリッド数
N_gy = int(L_y/np.sqrt(2)/Sigma_small)+1
L_gx = L_x/N_gx                      ###グリッドの長さ
L_gy = L_y/N_gy
T_min = (int(R_mask/2)*min(L_gx, L_gy)-2*Sigma_large)/3/V_max   ###マスク外衝突最小時間


mask = np.array([[i-int(R_mask/2),j-int(R_mask/2)] for i in range(R_mask) for j in range(R_mask)])
mask = np.delete(mask,int(R_mask**2/2), axis=0)   ####マスクの作成
N_mask = len(mask)   ###mask個数


###CBT
for i in range(1000):
    if N > 2**i:
        pass
    else:
        N_cbt = 2**i   ###cbt個数
        N_log = i      ###cbt更新数
        break
CBT_list = np.full((2*N_cbt-1),N,dtype=np.int64)
for i in range(N):
    CBT_list[N+i-1] = i
    
### 初期化    

t = 0.0    ###現在時刻
t_pos = 0.0   ###粒子の位置時刻
n_colision = 0   ###衝突回数
t_eepgm = 0.0
img = 0
event_list = np.full((N,N_mask), 9999999.0)
event_list_arg = np.full((N,N_mask), N, dtype=np.int64)
eepgm_list = np.full((N,2), N, dtype=np.int64)
eepgm_grid = np.full((N_gx,N_gy), N, dtype=np.int64)
colison_list = np.full(2*N_mask, N, dtype=np.int64)




###main

compute_eepgm(eepgm_grid, eepgm_list, pos)
compute_next_event_all(pos, vel, event_list, eepgm_grid, eepgm_list, event_list_arg, sigma_list)
single_event_list = np.min(event_list, axis=1)
single_event_list = np.append(single_event_list,999999999.0)
colision_arg = np.argmin(event_list, axis=1)
single_event_list_arg = event_list_arg[np.arange(N), colision_arg]
compute_CBT_all(single_event_list, CBT_list)
next_event = single_event_list[CBT_list[0]] 
next_event_arg  = np.array([CBT_list[0], single_event_list_arg[CBT_list[0]]])



#######出力#######
#calculate_msd_and_ngp_and_f(pos_msd_and_ngp, pos_0)
#calculate_msd_and_ngp(pos_msd_and_ngp, pos_0, msdout, ngpout)
snapshot(t, pos, sigma_list)
##################



ts = time.perf_counter()
print(ts)



for setps in range(n_steps):
    if dt:
        next_t = t + dt
    else:
        next_t = t + next_event
        
    while t_pos + next_event <= next_t:
        t = t_pos + next_event
        n_colision +=1
        
        compute_colision_pair(next_event, pos, vel, next_event_arg, pos_msd_and_ngp)
        
        
        if next_event > T_min:
            compute_pos_all(next_event, pos, vel, pos_msd_and_ngp, eepgm_grid)
            t_pos = t
            compute_eepgm(eepgm_grid, eepgm_list, pos)
            compute_next_event_all(pos, vel, event_list, eepgm_grid, eepgm_list, event_list_arg, sigma_list)
            single_event_list = np.min(event_list, axis=1)
            single_event_list = np.append(single_event_list,999999999.0)
            colision_arg = np.argmin(event_list, axis=1)
            single_event_list_arg = event_list_arg[np.arange(N), colision_arg]
            compute_CBT_all(single_event_list, CBT_list)
            next_event = single_event_list[CBT_list[0]] 
            """
            if next_event < 0:
                print('overlap_occured_while_if')
                break
            """
            next_event_arg  = np.array([CBT_list[0], single_event_list_arg[CBT_list[0]]])

            
        else:
            compute_next_event(pos, vel, event_list, eepgm_grid, eepgm_list, colison_list, next_event_arg, sigma_list, next_event)
            compute_CBT(colison_list, single_event_list, single_event_list_arg, event_list, \
                        event_list_arg, CBT_list, next_event_arg)
            next_event = single_event_list[CBT_list[0]] 
            """
            if next_event < 0:
                print('overlap_occured_while_else')
                break
            """
            next_event_arg  = np.array([CBT_list[0], single_event_list_arg[CBT_list[0]]])
    """
    if next_event < 0:
        print('overlap_occured_for1')
        break
    """        
    remain_t = next_t - t_pos
    compute_pos_all(remain_t, pos, vel, pos_msd_and_ngp, eepgm_grid)
    t = t_pos +remain_t
    t_pos = t
    compute_eepgm(eepgm_grid, eepgm_list, pos)
    compute_next_event_all(pos, vel, event_list, eepgm_grid, eepgm_list, event_list_arg, sigma_list)
    single_event_list = np.min(event_list, axis=1)
    single_event_list = np.append(single_event_list,999999999.0)
    colision_arg = np.argmin(event_list, axis=1)
    single_event_list_arg = event_list_arg[np.arange(N), colision_arg]
    compute_CBT_all(single_event_list, CBT_list)
    next_event = single_event_list[CBT_list[0]]
    """
    if next_event < 0:
        print('overlap_occured_for2')
        break
    """
    next_event_arg  = np.array([CBT_list[0], single_event_list_arg[CBT_list[0]]])
    
        ####出力#####
    print(t) 
#    snapshot(t, pos, sigma_list)
#    if setps == 199999:
#        pos_0 = deepcopy(pos_msd_and_ngp)
#    if setps >= 200000:
#        calculate_msd_and_ngp_and_f(pos_msd_and_ngp, pos_0)    
#    calculate_msd_and_ngp_and_f(pos_msd_and_ngp, pos_0)
#    calculate_msd_and_ngp(pos_msd_and_ngp, pos_0, msdout, ngpout)

#    if setps == n_steps-1:
#        snapshot(t, pos, sigma_list)
    #############
    
    
te = time.perf_counter()
print(te-ts)
print(n_colision)  


"""
   ###GIF作成###
images = []
for i in range(n_steps+1):
        file_name = snapshotout +'/'+str(i)+'.png'
        im = Image.open(file_name)
        images.append(im)
    
images[0].save(output_dir + '/animation' + '/' + senyuritu + 'check.gif', save_all=True, append_images=images[1:], loop=0, duration=50)
"""
