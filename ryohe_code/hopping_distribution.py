# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:31:01 2023

@author: 81905
"""
import numpy as np
import matplotlib.pyplot as plt

a = 1.4
pi = np.pi
file_number='2'
par_occ='0790'
pack = float(par_occ)/1000
par_occ1 = str(pack)
Lx = 64
Ly = Lx
LX = Lx
LY = Ly

loaded_file='./actual_dataset_updated_ver3/'+par_occ+'/'+file_number+'.npz'  
file=np.load(loaded_file, allow_pickle=True)

NN_number1=file['NN_number1']
NN_number2=file['NN_number2']
NN_number3=file['NN_number3']
hop_list=file['hop_list']
initial_pos=file['initial_position']
print(initial_pos[4091])



"""
for i in range(4096):
    if hop_list[i] < 0.01:
        print(str(i)+':'+str(hop_list[i]))
        print(NN_number1[i])
        print(NN_number2[i])
        print(NN_number3[i])
"""
 
"""       
plt.hist(hop_list)
plt.savefig("./hopping_hist/0790/10.png")
"""




        

