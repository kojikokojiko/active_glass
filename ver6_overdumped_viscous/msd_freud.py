import sys
import os
sys.path.append('/home/isobelab2022/build3/hoomd')

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import gsd.hoomd
import hoomd
import sys
from PIL import Image
import os
import math
import  numpy as np
import seaborn as sns
import sys
import math
import freud

nu=float(sys.argv[1])
pl=float(sys.argv[2])
fixed_percent=float(sys.argv[3])
ver=str(nu)+"_"+str(pl)+"_"+str(fixed_percent)


main_dir="./"+ver
temp_dir="./"+ver+"/log_pos_"+ver+".gsd"
traj = gsd.hoomd.open(temp_dir, 'rb')

pos=[]
for frame in traj:
    pos.append(frame.particles.position)
# pos=traj.particles.position


image=[]
for i in range(len(traj)):
    image.append(traj[i].particles.image)

box=freud.box.Box(Lx=64,Ly=64)
# msd_analyzer=freud.msd.MSD(mode='direct')
msd_analyzer=freud.msd.MSD(mode='direct',box=box)
msd_analyzer.compute(pos,images=image)
# msd_analyzer.compute(pos)
print(msd_analyzer.msd)
print(len(msd_analyzer.msd)) 
msd=msd_analyzer.msd
plt.plot(msd)
plt.savefig(main_dir+"/msd.png")
msd_analyzer.plot()