import sys
import os
sys.path.append('/home/isobelab2022/build3/hoomd')
import itertools
import math

import gsd.hoomd
import hoomd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from numba import jit, f8, i8, b1, void,njit
import matplotlib.patches as pat
import sys
import os
import fresnel
from PIL import Image
import IPython
import packaging.version
import random
import time
 

first_time=time.time()
def initialize_ra():



    return rx,ry

nu=float(sys.argv[1])
pl=float(sys.argv[2])



ver=str(nu)+"_"+str(pl)


main_dir="./"+ver
if not os.path.exists(main_dir): os.makedirs(main_dir)


data_file="../data/nu{0}.npz".format(nu)

npz=np.load(data_file)

rx=npz["rx"]
ry=npz["ry"]
sigma=npz["sigma"]



# print(sigma)
# Figureを作成
fig, ax = plt.subplots(figsize=(15,15))

# 軸の範囲を設定
ax.set_xlim([0, lx])  # x軸の範囲を0から5に設定
ax.set_ylim([0, ly])  # y軸の範囲を0から5に設定


# 複数の配列をまとめて一つのファイルに保存する方法
# https://note.nkmk.me/python-numpy-savez-savez-compressed/
# np.savez('test.npz', a=a, b=b, c=c)

from matplotlib.patches import Circle

for i in range(len(rx)):
    # Circleパッチを作成
    if (sigma[i]>0.3):
        c="r"
    else:
        c="b"

    circle = Circle((rx[i], ry[i]), sigma[i], color=c)
    # CircleパッチをAxesに追加
    ax.add_patch(circle)

ax.set_title("nu={0}".format(nu))

plt.savefig("nu{0}.png".format(nu))

print("OK")