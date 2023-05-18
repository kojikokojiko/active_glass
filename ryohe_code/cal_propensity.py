# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:39:52 2023

@author: 81905
"""


import numpy as np


###各物理的parameter
Mol_small, Mol_large = 2, 1 ###モル比
Sigma_ratio = 1.4
Density = 0.78     ###占有率
Elast = 1.0     ###反発係数
Sigma_small = np.sqrt(Density*64*64*(Mol_large+Mol_small)/4096/np.pi/(Mol_large*Sigma_ratio**2 + Mol_small))   ###粒子（小）半径
Sigma_large = Sigma_ratio * Sigma_small
L_x = 64
L_y = L_x

###読み込みに関わるparameter
Density_str = '0780'
config_number = 10
config_number = str(config_number)

###hoplistの計算
hop_distance=0.7*Sigma_small*2
hoplist = np.full(4096, 0)
ini_pos_file = './dataset/'+Density_str+'/initial_pos/'+Density_str+'_'+config_number+'.dat'

table_inipos = np.loadtxt(ini_pos_file)
for i in range(100):
    pos = './dataset/'+Density_str+'/pos/DIS_'+config_number+'/dis_over'+str(i+1)+'.dat'
    table_pos = np.loadtxt(pos)
    for j in range(4096):
        del_x = table_pos[j][0]-table_inipos[j][0]
        del_y = table_pos[j][1]-table_inipos[j][1]
        if del_x > L_x / 2:
            del_x -=L_x
        elif del_x < -L_x / 2:
            del_x +=L_x
        if del_y > L_y / 2:
            del_y -=L_y
        elif del_y < -L_y / 2:
            del_y +=L_y
        delr = np.sqrt( del_x**2 + del_y**2 )
        if delr > hop_distance:
            hoplist[j] += 1 
        else:
            pass
hoplist = hoplist/100


f_r='./dataset/'+Density_str+'/hop_list/'+config_number+'.dat'
with open(f_r, 'w') as f0:
    for i in range(4096):
        f0.write(str(hoplist[i])+' ')

