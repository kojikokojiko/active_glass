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
# INPUT PARAMETER
nu=float(sys.argv[1])
pl=float(sys.argv[2])

 
# this section is fixed  #############
#particle parameter
Nx = 64
Ny = 64
#粒子数
N = Nx*Ny 

lx = 64
ly = lx

#AL  半径比
AL = 1.4
EL = 1/3 #個数比
N1 = int(N*EL) #大粒子の個数　
N0 = N-N1 #小粒子の個数
##############################



kbT=0.01
m=1.0
epsilon=1.0



v0=1.0
tau=pl/v0
mu=1.0 #mobility

gamma=1.0/mu
DR=1.0/tau
gamma_r=kbT/DR




# 刻み幅小さすぎの可能性もあるから大きめにしてみてもいいかも

real_time=80
dt = 4.0e-5
nsteps=int(real_time/dt)
pos_hout=int(nsteps/500)
# thermo_hout=int(nsteps/2.0)

data_file="../data/nu{0}.npz".format(nu)

npz=np.load(data_file)

rx=npz["rx"]
ry=npz["ry"]
rx=rx-lx/2.0
ry=ry-ly/2.0
sigma=npz["sigma"]
typeid=npz["typeid"]
print(N)

set_sigma=list(set(sigma))
small_sigma=set_sigma[0]
large_sigma=set_sigma[1]


sigma_ss=small_sigma
sigma_ll=large_sigma
sigma_sl=(small_sigma+large_sigma)/2



ver=str(nu)+"_"+str(pl)
main_dir="./"+ver
if not os.path.exists(main_dir): os.makedirs(main_dir)
print(main_dir)
os.chdir(main_dir)



position=[]
for i in range(len(rx)):
    position.append((rx[i],ry[i],0.0))



snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N
snapshot.particles.position = position[0:N]
orientation = np.zeros((N,4))

for i in range(N):
    theta=random.uniform(-np.pi,np.pi)
    orientation[i][0]=math.cos(theta/2)
    orientation[i][3]=math.sin(theta/2)

# (1,0,0,0,)
snapshot.particles.orientation = orientation
snapshot.particles.typeid = (N)*[0]
snapshot.particles.types = ['small','large']
snapshot.particles.diameter=sigma
snapshot.particles.mass = np.ones((N))
snapshot.configuration.box = [lx, ly, 0, 0, 0, 0]



sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=12)

sim.create_state_from_snapshot(snapshot)
# Integration information


cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell, mode="shift")
lj.params[("small", "small")] = dict(epsilon=epsilon, sigma=sigma_ss)
lj.r_cut[("small", "small")] = 2**(1/6)*sigma_ss


lj.params[("small", "large")] = dict(epsilon=epsilon, sigma=sigma_sl)
lj.r_cut[("small", "large")] = 2**(1/6)*sigma_sl

lj.params[("large", "large")] = dict(epsilon=epsilon, sigma=sigma_ll)
lj.r_cut[("large", "large")] = 2**(1/6)*sigma_ll

# rotational_diffusion = 20.0
# Apply Stokes-Einstein
# traslational_diffusion = 3.0 * rotational_diffusion

#???????
brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kbT)
brownian.gamma.default = gamma
brownian.gamma_r.default =[gamma_r,gamma_r,0]
# brownian.gamma_r.default = np.full((3,), ktemp / rotational_diffusion)
active = hoomd.md.force.Active(filter=hoomd.filter.All())

act_force = v0 * brownian.gamma.default
active.active_force["large"] = (act_force, act_force,0)
active.active_torque["large"] = (0, 0, 0)
active.active_force["small"] = (act_force, act_force,0)
active.active_torque["small"] = (0, 0, 0)
rotational_diffusion_updater = active.create_diffusion_updater(
 trigger=hoomd.trigger.Periodic(1), rotational_diffusion=DR
)

integrator = hoomd.md.Integrator(
 dt=dt,
 methods=[brownian],
 forces=[lj,active],
)

# velocity_operation=hoomd.update.CustomUpdater(action=RelativeFlow(ave_flow,dt),trigger=1)
# sim.operations+=velocity_operation
sim.operations += rotational_diffusion_updater
sim.operations.integrator = integrator

# カスタムクラス
class PrintTimestep(hoomd.custom.Action):
    def act (self,timestep):
        print(timestep)
custom_action = PrintTimestep()
custom_op = hoomd.write.CustomWriter(action=custom_action,
                                 trigger=hoomd.trigger.Periodic(10000))
sim.operations.writers.append(custom_op)





# logger定義
pos_logger = hoomd.logging.Logger()
# pos_logger.add(sim, quantities=['timestep', 'walltime'])
pos_logger.add(lj ,quantities=["forces"])
# pos_logger.add(thermodynamic_properties,quantities=["kinetic_temperature","kinetic_energy","potential_energy","volume"])
gsd_writer_pos = hoomd.write.GSD(filename="log_pos_"+ver+".gsd",
                             trigger=hoomd.trigger.Periodic(pos_hout),
                             mode='xb',
                             filter=hoomd.filter.All())
gsd_writer_pos.log = pos_logger

sim.operations.writers.append(gsd_writer_pos)

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kbT)

sim.run(0)

print("-----run--------")
second_time=time.time()

sim.run(nsteps)

print(time.time()-first_time)

print(time.time()-second_time)
os.chdir("../")



traj = gsd.hoomd.open('./'+ver+'/log_pos_'+ver+'.gsd', 'rb')


# traj = gsd.hoomd.open('log_force2d_'+ver+'.gsd', 'rb')


output_dir=main_dir+"/figure_2d"
if not os.path.exists(output_dir): os.makedirs(output_dir)
figsize=(10,10)
plt.figure(figsize=figsize,dpi=150)

print(len(traj))
sigma=traj[0].particles.diameter
set_sigma=list(set(sigma))

for t in range(len(traj)-1,len(traj)-6,-2):
    print(t)
    bx=plt.axes()
    plt.axis([-lx/2,lx/2,-ly/2,ly/2])
    position=traj[t].particles.position
   
    for i in range(N):
        # print(i)

            # Circleパッチを作成
        if (sigma[i]==set_sigma[0]):
            c="r"
        else:
            c="b"

        c=pat.Circle(xy=(position[i][0],position[i][1]),radius=sigma[i]/2,fc=c)
        bx.add_patch(c)

    plt.title("step"+str(t))
    plt.savefig(output_dir+"/figure{0}.png".format(t))
    plt.cla()
    plt.clf()

    ############アニメーション################    
images=[]
# image_num=sum(os.path.isfile(os.path.join(pic_output name)) for name in os.listdir(pic_output))
image_num=sum(os.path.isfile(os.path.join(output_dir,name))for name in os.listdir(output_dir))
print(image_num)
for i in range(0,image_num):
    file_name=output_dir+"/figure"+str(i)+".png"
    im=Image.open(file_name)
    images.append(im)

gif_output_dir=main_dir+"/abpgif2"

if not os.path.exists(gif_output_dir): os.makedirs(gif_output_dir)
images[0].save(gif_output_dir+"/out_ela2.gif",save_all=True,append_images=images[1:],loop=0,duration=10)
    
    