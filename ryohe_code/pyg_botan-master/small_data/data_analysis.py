# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:29:54 2022

@author: 81905
"""

import numpy as np

loaded_data = './T0.44/test/isoconfig_N4096T0.44_401_tc07.npz'
table = np.load(loaded_data)

print(list(table))

print(table['types'])
print(table['initial_positions'])
print(table['positions'])