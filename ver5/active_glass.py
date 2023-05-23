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
import pickle
 

media_dir="/media/isobelab2022/data/active_glass/ver2"
first_time=time.time()
# INPUT PARAMETER
nu=float(sys.argv[1])
fixed_percent=float(sys.argv[2])
kbT=float(sys.argv[3])
pl=float(sys.argv[4])


 

random.seed(12)
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


m=1.0
epsilon=1.0



v0=1.0
tau=pl/v0
mu=1.0 #mobility

gamma=1.0/mu
DR=1.0/tau
gamma_r=kbT/DR

# 刻み幅小さすぎの可能性もあるから大きめにしてみてもいいかも

# 総時間幅
real_time=150
# 出力を始めるまでの時間幅＝＞平衡状態に達するまでの時間
after_time=50
# 出力をしている時間幅
pos_out_time=(real_time-after_time)
# 刻み時間
dt = 4.0e-5
# 総ステップ数
nsteps=int(real_time/dt)
# 出力を始めるまでのステップ数
after_steps=int(after_time/dt)
# 出力をしているときの送ステップ数
pos_out_steps=int(pos_out_time/dt)

########################################################出力間隔調整
# # 出力をしているときのステップ数の間隔
# pos_out_steps_period=5
# # 総出力回数
# num_out=int(pos_out_steps/pos_out_steps_period)
########################################################総出力回数調整
# 総出力回数
num_out=int(500)
# 出力をしているときのステップ数の間隔
pos_out_steps_period=int(pos_out_steps/num_out)
##############################################################
# image出力枚数
image_out_num=150
# 総出力回数に対してイメージを出力させる間隔
image_out_period=int(num_out/image_out_num)




# pos_hout=int(nsteps/500)
# thermo_hout=int(nsteps/2.0)

data_file="../data/nu0.8.npz"

npz=np.load(data_file)

rx=npz["rx"]
ry=npz["ry"]
rx=rx-lx/2.0
ry=ry-ly/2.0
# sigma=npz["sigma"]
# typeid=npz["typeid"]
print(N)



temp_radius = np.sqrt(lx*ly*nu/((N0+N1*AL**2)*np.pi)) #radius

def binary_a(radius,N,ly,N1,AL):
    sigma = [0 for i in range(N)]
    N1_num=0
    for i in range(0,N-1,2):
        K = int(i/ly)+1
        if K%2==1:
            sigma[i] = radius
            if N1_num<=N1:
                sigma[i+1]=AL*radius
                N1_num+=1
            else:
                sigma[i+1]=radius
        
        else:
            if N1_num<=N1:
                sigma[i]=AL*radius
                N1_num+=1
            else:
                sigma[i]=radius
            
            sigma[i+1]=radius
    return sigma
         


sigma=binary_a(temp_radius,N,ly,N1,AL)
print(len(sigma))
sigma=[sigma[i]*2 for i in range(N)]
print(len(sigma))



set_sigma=list(set(sigma))
small_sigma=set_sigma[0]
large_sigma=set_sigma[1]

typeid=[0 for i in range(N)]
for i in range(N):
    if sigma[i]==small_sigma:
        typeid[i]=0
    else:
        typeid[i]=1


sigma_ss=small_sigma
sigma_ll=large_sigma
sigma_sl=(small_sigma+large_sigma)/2



ver=str(nu)+"_"+str(fixed_percent)+"_"+str(kbT)+"_"+str(pl)

main_dir="./"+ver
if not os.path.exists(main_dir): os.makedirs(main_dir)
print(main_dir)
os.chdir(main_dir)



traj_dir=media_dir+"/"+ver
if not os.path.exists(traj_dir):
    os.makedirs(traj_dir)

traj_path=traj_dir+"/log_pos_"+ver+".gsd"

fixed_id=[]

fixed_N=int(N*fixed_percent)
move_N=N-fixed_N


while len(fixed_id)<fixed_N:
    id=random.randint(0,N-1)
    if id not in fixed_id:
        fixed_id.append(id)

move_id=[num for num in range(N) if num not in fixed_id]

print(len(move_id))

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

snapshot.particles.orientation = orientation
snapshot.particles.typeid = typeid
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


#???????
brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.Tags(move_id), kT=kbT)
brownian.gamma.default = gamma
# brownian.gamma_r.default =[gamma_r,gamma_r,0]
# brownian.gamma_r.default = np.full((3,), ktemp / rotational_diffusion)

active = hoomd.md.force.Active(filter=hoomd.filter.Tags(move_id))
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
# pos_logger.add(thermodynamic_properties,quantities=["kinetic_temperature","kinetic_energy","potential_energy","volume"])
gsd_writer_pos = hoomd.write.GSD(filename=traj_path,
                            #  trigger=hoomd.trigger.Periodic(pos_hout),
                            trigger=hoomd.trigger.And([
                                    hoomd.trigger.After(after_steps),
                                    hoomd.trigger.Periodic(pos_out_steps_period)]),
                             mode='xb',
                             filter=hoomd.filter.All())
gsd_writer_pos.log = pos_logger

sim.operations.writers.append(gsd_writer_pos)

sim.state.thermalize_particle_momenta(filter=hoomd.filter.Tags(move_id), kT=kbT)

sim.run(0)

print("-----run--------")
second_time=time.time()



variables = {
    'real_time': real_time,
    'after_time': after_time,
    'pos_out_time': pos_out_time,
    'dt': dt,
    'nsteps': nsteps,
    'after_steps': after_steps,
    'pos_out_steps': pos_out_steps,
    'pos_out_steps_period': pos_out_steps_period,
    'num_out': num_out,
    'image_out_num': image_out_num,
    'image_out_period': image_out_period,
    'm': m,
    'epsilon': epsilon,
    'kbT': kbT,
    'nu': nu,
    'fixed_percent': fixed_percent,
    'sigma_ss': sigma_ss,
    'sigma_ll': sigma_ll,
    'sigma_sl': sigma_sl,
    'pl': pl,
    'v0': v0,
    'tau': tau,
    'mu': mu,
    'gamma': gamma,
    'DR': DR,
    'gamma_r': gamma_r,
}



# ファイルにほぞん.txt
with open(traj_dir+'/data.txt', 'w') as f:
    for var_name, var_value in variables.items():
        f.write(f'{var_name} {var_value}\n')


# ファイルに保存.pickle
with open(traj_dir+'/data.pickle', 'wb') as handle:
    pickle.dump(variables, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # ファイルから読み込み
# with open('my_dict.pickle', 'rb') as handle:
#     loaded_dict = pickle.load(handle)

sim.run(nsteps)

print(time.time()-first_time)

print(time.time()-second_time)
os.chdir("../")



traj = gsd.hoomd.open(traj_path, 'rb')






# traj = gsd.hoomd.open('log_force2d_'+ver+'.gsd', 'rb')


output_dir=main_dir+"/figure_2d"
if not os.path.exists(output_dir): os.makedirs(output_dir)
figsize=(10,10)
plt.figure(figsize=figsize,dpi=150)

print(len(traj))
sigma=traj[0].particles.diameter
set_sigma=list(set(sigma))

for t in range(len(traj)-1,0,image_out_period):
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
        if i<100:
            plt.annotate(str(i),(position[i][0],position[i][1]))
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
    
    