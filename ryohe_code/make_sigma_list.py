# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:05:30 2023

@author: 81905
"""

import numpy as np

Mol_small, Mol_large = 2, 1 ###モル比
Sigma_ratio = 1.4
Density = 0.78     ###占有率
Sigma_small = np.sqrt(Density*64*64*(Mol_large+Mol_small)/4096/np.pi/(Mol_large*Sigma_ratio**2 + Mol_small))   ###粒子（小）半径
Sigma_large = Sigma_ratio * Sigma_small

sigma_list = np.full(4096, Sigma_large)
sigma_decision = []
with open('sigma_init' + '.txt') as f:
    for line in f:
        sigma_decision_element = float(line)
        sigma_decision.append(sigma_decision_element)
for i in range(4096):
    if sigma_decision[i] < 0.1:
        sigma_list[i] = Sigma_small
        
f_in='./dataset/0780/sigma_list.dat'
with open(f_in, 'w') as f1:
    for i in range(4096):
        f1.write(str(sigma_list[i])+' ')