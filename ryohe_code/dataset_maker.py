# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:19:00 2023

@author: 81905
"""

import numpy as np

###各ファイルとデータの読み込み、読み込み時のパラメーター変え忘れ注意!!!
main_directory = '0800'###占有率
file_number = str(10)

ini_pos_f = './dataset/'+main_directory+'/initial_pos/'+main_directory+'_'+file_number+'.dat'
sig_list_f = './dataset/'+main_directory+'/sigma_list.dat'
NNnum1_f = './dataset/'+main_directory+'/NN1_list/'+file_number+'.dat'
NNnum2_f = './dataset/'+main_directory+'/NN2_list/'+file_number+'.dat'
NNnum3_f = './dataset/'+main_directory+'/NN3_list/'+file_number+'.dat'
cut_f = './dataset/'+main_directory+'/cutoff/'+file_number+'.dat'
re_cut_f = './dataset/'+main_directory+'/re_cutoff/'+file_number+'.dat'
fv_f = './dataset/'+main_directory+'/free_vol/'+file_number+'.dat'
re_fv_f = './dataset/'+main_directory+'/re_free_vol/'+file_number+'.dat'
fv_sum_f = './dataset/'+main_directory+'/free_vol_sum/'+file_number+'.dat'
re_fv_sum_f = './dataset/'+main_directory+'/re_free_vol_sum/'+file_number+'.dat'
fv_ave_f = './dataset/'+main_directory+'/free_vol_ave/'+file_number+'.dat'
re_fv_ave_f = './dataset/'+main_directory+'/re_free_vol_ave/'+file_number+'.dat'
fs_f = './dataset/'+main_directory+'/free_sur/'+file_number+'.dat'
local_pre_f = './dataset/'+main_directory+'/local_pressure/'+file_number+'.dat'
hop_list_f = './dataset/'+main_directory+'/hop_list/'+file_number+'.dat'

#========================ver2追加分================================
fv_NN_f = './dataset/'+main_directory+'/fv_NN/'+file_number+'.npy'
angle_f = './dataset/'+main_directory+'/angle/'+file_number+'.npy'
#==================================================================

ini_pos = np.loadtxt(ini_pos_f)
sig_list = np.loadtxt(sig_list_f)
NNnum1 = np.loadtxt(NNnum1_f)
NNnum2 = np.loadtxt(NNnum2_f)
NNnum3 = np.loadtxt(NNnum3_f)
cut = np.loadtxt(cut_f)
re_cut = np.loadtxt(re_cut_f)
fv = np.loadtxt(fv_f)
re_fv = np.loadtxt(re_fv_f)
fv_sum = np.loadtxt(fv_sum_f)
re_fv_sum = np.loadtxt(re_fv_sum_f)
fv_ave = np.loadtxt(fv_ave_f)
re_fv_ave = np.loadtxt(re_fv_ave_f)
fs = np.loadtxt(fs_f)
local_pre = np.loadtxt(local_pre_f)
hop_list = np.loadtxt(hop_list_f)

#======================ver2追加分============================
fv_NN=np.load(fv_NN_f, allow_pickle=True)
angle=np.load(angle_f, allow_pickle=True)
#============================================================

#np.savez('./actual_dataset/'+main_directory+'/'+file_number, initial_position=ini_pos, sigma_list=sig_list, NN_number1=NNnum1, NN_number2=NNnum2, NN_number3=NNnum3, cutoff=cut, re_cutoff=re_cut, free_volume=fv, re_free_volume=re_fv, free_volume_sum=fv_sum, re_free_volume_sum=re_fv_sum, free_volume_ave=fv_ave, re_free_volume_ave=re_fv_ave, free_surface=fs, local_pressure=local_pre, hop_list=hop_list)

#======================ver2追加分============================
np.savez('./actual_dataset_updated_ver3/'+main_directory+'/'+file_number, initial_position=ini_pos, sigma_list=sig_list, NN_number1=NNnum1, NN_number2=NNnum2, NN_number3=NNnum3, cutoff=cut, re_cutoff=re_cut, free_volume=fv, re_free_volume=re_fv, free_volume_sum=fv_sum, re_free_volume_sum=re_fv_sum, free_volume_ave=fv_ave, re_free_volume_ave=re_fv_ave, free_surface=fs, local_pressure=local_pre, hop_list=hop_list, fv_NN=fv_NN, angle=angle)
#============================================================
