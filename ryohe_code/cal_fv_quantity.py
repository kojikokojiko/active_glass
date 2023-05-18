# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:23:30 2023

@author: 81905
"""

import numpy as np

###ファイル読み込みに関するparameter
Density_str = '0800'
file_num = 10
file_num = str(file_num)

###近接粒子番号読み込み
NNnum1_file = './dataset/'+Density_str+'/NN1_list/'+file_num+'.dat'
NNnum2_file = './dataset/'+Density_str+'/NN2_list/'+file_num+'.dat'
NNnum3_file = './dataset/'+Density_str+'/NN3_list/'+file_num+'.dat'

#NNnum1 = np.loadtxt(NNnum1_file)#←このままだとfloat型で、後の配列番号指定するときにerrorが出る
NNnum1 = np.loadtxt(NNnum1_file, dtype='int')#np.loadtxt()の引数で型の指定が可能
NNnum2 = np.loadtxt(NNnum2_file, dtype='int')
NNnum3 = np.loadtxt(NNnum3_file, dtype='int')


###自由体積読み込み
fv_list_file = './dataset/'+Density_str+'/free_vol/'+file_num+'.dat'
fv_list = np.loadtxt(fv_list_file)

###相対自由体積読み込み
re_fv_list_file = './dataset/'+Density_str+'/re_free_vol/'+file_num+'.dat'
re_fv_list = np.loadtxt(re_fv_list_file)

###各近接粒子まで考慮した自由体積、相対自由体積の積算、平均の算出
fv_sum = np.full((4096,3),0.0)
fv_ave = np.full((4096,3),0.0)
re_fv_sum = np.full((4096,3),0.0)
re_fv_ave = np.full((4096,3),0.0)

for i in range(4096):
###0列目に加える操作、counterが積算の計算に含まれた粒子の数でaveは積算をそれで割ればよい
###初めに注目する粒子自身の値を加えて、counterを1とする。その後-1以外の所を順当に加え、counterを1ずつ増やす。
    fv_sum[i][0] += fv_list[i]
    re_fv_sum[i][0] += re_fv_list[i]
    counter = 1
    for j in range(10):
        if NNnum1[i][j] > -0.5:
            fv_sum[i][0] += fv_list[NNnum1[i][j]]
            re_fv_sum[i][0] += re_fv_list[[NNnum1[i][j]]]
            counter += 1
        else:
            pass
    fv_ave[i][0] = fv_sum[i][0]/counter
    re_fv_ave[i][0] = re_fv_sum[i][0]/counter 
    
###1列目に加える操作：まず、第1近接粒子までのcounterとsumの情報を引き継ぐ、後は同様に
    fv_sum[i][1] += fv_sum[i][0]
    re_fv_sum[i][1] += re_fv_sum[i][0]
    for k in range(20):
        if NNnum2[i][k] > -0.5:
            fv_sum[i][1] += fv_list[NNnum2[i][k]]
            re_fv_sum[i][1] += re_fv_list[[NNnum2[i][k]]]
            counter += 1
        else:
            pass
    fv_ave[i][1] = fv_sum[i][1]/counter
    re_fv_ave[i][1] = re_fv_sum[i][1]/counter
    
###2列目に加える操作
    fv_sum[i][2] += fv_sum[i][1]
    re_fv_sum[i][2] += re_fv_sum[i][1]
    for l in range(32):
        if NNnum3[i][l] > -0.5:
            fv_sum[i][2] += fv_list[NNnum3[i][l]]
            re_fv_sum[i][2] += re_fv_list[[NNnum3[i][l]]]
            counter += 1
        else:
            pass
    fv_ave[i][2] = fv_sum[i][2]/counter
    re_fv_ave[i][2] = re_fv_sum[i][2]/counter

###最終的に出力するのは以下    
f_in='./dataset/'+Density_str+'/free_vol_sum/'+file_num+'.dat'
with open(f_in, 'w') as f1:
    for i in range(4096):
        for j in range(3):
            f1.write(str(fv_sum[i][j])+' ')
        f1.write('\n')
        
f_in='./dataset/'+Density_str+'/re_free_vol_sum/'+file_num+'.dat'
with open(f_in, 'w') as f2:
    for i in range(4096):
        for j in range(3):
            f2.write(str(re_fv_sum[i][j])+' ')
        f2.write('\n')  
        
f_in='./dataset/'+Density_str+'/free_vol_ave/'+file_num+'.dat'
with open(f_in, 'w') as f3:
    for i in range(4096):
        for j in range(3):
            f3.write(str(fv_ave[i][j])+' ')
        f3.write('\n')
        
f_in='./dataset/'+Density_str+'/re_free_vol_ave/'+file_num+'.dat'
with open(f_in, 'w') as f4:
    for i in range(4096):
        for j in range(3):
            f4.write(str(re_fv_ave[i][j])+' ')
        f4.write('\n')          


    
        
                   
                
