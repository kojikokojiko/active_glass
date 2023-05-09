# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:54:07 2022
Over damped Langevin Equetion

@author: nase0
"""
from tracemalloc import Snapshot
import numpy as np
import matplotlib .pyplot as plt
from PIL import Image
import matplotlib.patches as pat 
import os
import math
import sys

#------------------------------------------------------------parameter


#particle parameter
Nx = 64
Ny = 64
#粒子数
N = Nx*Ny 

lx = 64
ly = lx
#粒子占有率
nu = float(sys.argv[1])
print(nu)
#AL  半径比
AL = 1.4
EL = 1/3 #個数比
N1 = int(N*EL) #大粒子の個数　
N0 = N-N1 #小粒子の個数

M = 1.0 #mass
#M=m/zeta*tau_v dimensionless mass


pl=0.01
#持続長
v0=1
tau = pl/v0
mu = 1.0 #mobility
a = np.sqrt(lx*ly*nu/((N0+N1*AL**2)*np.pi)) #radius

A = 7*a#規格化定数

sigma = [0 for i in range(N)]
epsilon = 1.0

KBT = 0.01
TEMP = 1

eta = 1.0 #粘性率


#numerical integration(数値積分)
t = 5#total time
h = 0.001 #time step
#tout = 0.01 #interval time for output date




#(total/output) number of time step
nsteps =int(t/h)
#nout = int(tout/h)
#nsteps=1
nout = 75

#fn=int(t//tout) #file number
count=1





####変数初期値######
#rx = np.zeros(N)
#ry = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
# とある粒子にかかる力x,y
ax = np.zeros(N)
ay = np.zeros(N)
Pot = 0.0

rc=np.zeros(N)





#============================================高速化
aroundGridNum = 5#注目グリッドから左何個分参照するか
reusableCount = 0
select_index=0

"""
bo=(lx/0.3)
l_gx_d=lx/bo
l_gx=l_gx_d#グリッドの単位長さ
l_gy=l_gx
n_gx=math.ceil(lx/l_gx)#x方向のグリッド総数
n_gy=math.ceil(ly/l_gy)#y方向のグリッド総数
n_g_all=n_gx*n_gy
"""
n_gx = int(lx / (np.sqrt(2) * a) + 1)
n_gy = int(ly / (np.sqrt(2) * a) + 1)

n_gx = n_gx*2
n_gy = n_gy*2

#---
l_gx = lx / n_gx#グリッドのx方向の個数
l_gy = ly / n_gy#グリッドのy方向の個数
n_g_all = n_gx*n_gy#グリッドの数
l_gx_d = l_gx

neighbor_row=[]#隣接した粒子のxのリスト
neighbor_col=[]#隣接した粒子のyのリスト
neighbor_len=2*(aroundGridNum*(aroundGridNum+1))#グリッド参照配列長さ

def makeneighborList(aroundGridNum,neighbor_row,neighbor_col):
#xの近接リスト
    neighborRowValue = -aroundGridNum
  # countIndexRow = 0
    for i in range(aroundGridNum+1):
        for j in range(aroundGridNum):
      # neighbor_row[countIndexRow]=neighborRowValue
            neighbor_row.append(neighborRowValue)
      # countIndexRow+=1
        neighborRowValue+=1
  

    for i in range(aroundGridNum):
        for j in range(aroundGridNum+1):
      # neighbor_row[countIndexRow]=neighborRowValue
          neighbor_row.append(neighborRowValue)
      # countIndexRow+=1

        neighborRowValue+=1

# yの近接リスト
    neighborColValue=-aroundGridNum
    for i in range(aroundGridNum+1):
        for j in range(aroundGridNum):
            neighbor_col.append(neighborColValue)
            neighborColValue+=1
        neighborColValue=-aroundGridNum

    for i in range(aroundGridNum):
        for j in range(aroundGridNum+1):
            neighbor_col.append(neighborColValue)
            neighborColValue+=1
        neighborColValue=-aroundGridNum
    
    return neighbor_row,neighbor_col

neighbor_row,neighbor_col=makeneighborList(aroundGridNum,neighbor_row,neighbor_col)

#前の関数で作った隣接ペアリストの作成
def gmap_create2(N,rx,ry,l_gx,l_gy,n_gx,n_gy,neighbor_row,neighbor_col,neighbor_len,
                 lx,ly,aroundGridNum):
    
    # //グリッドとペアリストの初期化
    # G_MAP= np.zeros((N_GY,N_GX))
    G_MAP=np.array([[-1]*n_gx]*n_gy)
    # PAIRLIST=np.ones((N,10))*-1
    PAIRLIST=np.array([[-1]*20]*N)


    for i in range(N):
      gx_map=int(rx[i]/l_gx)
      # gy_map=n_gy-(aroundGridNum+1)-int(ry[i]/l_gy)
      gy_map=int(ry[i]/l_gy)

      G_MAP[gy_map][gx_map]=i
      
      #print(gx_map)

    for i in range(n_gy):
        for j in range(n_gx):
            particle_counter=0
            if (G_MAP[i][j]!=-1):
                select_index=G_MAP[i][j]
                for k in range(neighbor_len):
                    search_gx=j + neighbor_col[k]
                    search_gy=i + neighbor_row[k]
                    # グリッドの周期境界
                    if (search_gx>=n_gx):
                        search_gx-=n_gx
                    elif (search_gx<0):
                        search_gx+=n_gx
                    if (search_gy>=n_gy):
                        search_gy-=n_gy
                    elif (search_gy<0):
                        search_gy+=n_gy
                    if (G_MAP[search_gy][search_gx]!=-1):
                        PAIRLIST[select_index][particle_counter]=G_MAP[search_gy][search_gx]
                        particle_counter+=1
                        #print(particle_counter)
                    
                    PAIRLIST[select_index][-1]=particle_counter
            
    return PAIRLIST
#=============================================-=WCAポテンシャルの設定
def force(ax,ay,Pot,N,rx,ry,vx,vy,lx,ly,PAIRLIST):
    
    
   
    for i in range(N):
        ax[i] = 0.0
        ay[i] = 0.0
    Pot=0.0
    Pot_ij = 0.0
    
    for i in range(N):
        roop_num=PAIRLIST[i][-1]
        # print(roop_num)
        for j in range(roop_num):
            
            rxij=rx[i]-rx[PAIRLIST[i][j]]
            #  Xのminimum image convention
            if (rxij>=lx/2):
                rxij=rxij-lx
            elif rxij<-lx/2:
                rxij=rxij+lx
            #else:
            #rxij=rxij
         #  Yのminimum image convention
            ryij=ry[i]-ry[PAIRLIST[i][j]]
            if (ryij>=ly/2):
                ryij=ryij-ly
            elif ryij<-ly/2:
                ryij=ryij+ly
            #else:
                #ryij=ryij
           
            r2=rxij*rxij+ryij*ryij#粒子間距離の2乗
            #if r2<0.25:#重なり合うときの補正
                #r2 = 0.25
            s_ij=sigma[i]+sigma[select_index]
            rc= 2**(1.0/6)*s_ij
            #print(rc)
            rc2=rc*rc
            s_ij2=s_ij**2
            ir2=s_ij2/r2
            ir6=ir2*ir2*ir2
            ir12=ir6*ir6
           

            #WCAポテンシャルの条件
            if(r2>=rc2):
                fx=0.0
                fy=0.0
                Pot_ij=0.0
            else:
                
                f_facter = 24.0*epsilon*(2.0*ir12-ir6)*ir2
                fx=f_facter*rxij
                fy=f_facter*ryij
                Pot_ij=4.0*(ir12-ir6)+1.0
                
               
            ax[i]+=fx
            ay[i]+=fy
            ax[PAIRLIST[i][j]]-=fx
            ay[PAIRLIST[i][j]]-=fy
            Pot+=Pot_ij
    return ax,ay,Pot

#Langevin熱浴を設定
def noise(ax,ay,N,DL):
    for i in range(N):
        ax[i]=ax[i]+DL*np.random.normal()
        ay[i]=ay[i]+DL*np.random.normal()

#def activeの要素を入れる
theta=np.zeros(N)
def active(ax,ay,N,theta):
    for i in range(N):
        eta_j = np.random.normal(0,1)
        theta[i] = theta[i]+np.sqrt(2/tau)*eta_j*h
        #print(theta[i])
        
        ax[i] = ax[i]+v0*np.cos(theta[i])
        ay[i] = ay[i]+v0*np.sin(theta[i])  


def binary_a(a,sigma,N,ly,N1,AL):
    N1_num=0
    for i in range(0,N-1,2):
        K = int(i/ly)+1
        if K%2==1:
            sigma[i] = a
            if N1_num<=N1:
                sigma[i+1]=AL*a
                N1_num+=1
            else:
                sigma[i+1]=a
        
        else:
            if N1_num<=N1:
                sigma[i]=AL*a
                N1_num+=1
            else:
                sigma[i]=a
            
            sigma[i+1]=a
         
            
image_dir= "figure_binary_pl0.01_0.55"
 
if not os.path.exists(image_dir):
# ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(image_dir)


k=0



#-------------------------------------------------------------
#初期配置
"""
for i in range(Nx):
    for j in range(Ny):
        rx[k]=dx+i*2*dx
        ry[k]=dy+j*2*dy
        k+=1
"""
binary_a(a,sigma,N,ly,N1,AL)
#sigma=np.full(N,0.5)

#初期配置データ読み込み
# data_file="../position_set/N0064_0550_033.dat"
data_file='/home/isobelab2022/projects/active_glass/inaba_file/position_set/N0064_0'+str(int(nu*1000))+'_033.dat'
# projects/active_glass/inaba_file/position_set/N0064_0550_033.dat
rx=np.loadtxt(data_file,comments="#" ,usecols=0)
ry=np.loadtxt(data_file,comments="#" , usecols=1)


typeid=np.zeros(N)
set_radius=list(set(sigma))
print(set_radius)
print(set_radius[0])

for i in range(N):
    if sigma[i]==set_radius[0]:
        typeid[i]=0
    elif sigma[i]==set_radius[1]:
        typeid[i]=1

print(typeid)




print(sigma[0])
sigma=[n*2 for n in sigma]
print(sigma[0])



np.savez('nu{0}.npz'.format(nu), rx=rx, ry=ry, sigma=sigma,typeid=typeid)

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
    if (sigma[i]>set_radius[0]*2):
        c="r"
    else:
        c="b"

    circle = Circle((rx[i], ry[i]), sigma[i]/2, color=c)
    # CircleパッチをAxesに追加
    ax.add_patch(circle)
    # ax.annotate(typeid[i],xy=(rx[i],ry[i]))


ax.set_title("nu={0}".format(nu))

plt.savefig("nu{0}.png".format(nu))

print("OK")
















with open('DE/'+str(Nx)+'/'+str(int(nu*1000))+'/analy_data/f_info.dat') as f:
    for i in range(12):
        next(f)
    l=f.readline()
    l=l.split()
    ae=float(l[2])/2
    
#ゆらぎ
DL = KBT/6*np.pi*eta*ae

#初期速度
np.random.seed(1)
for i in range(N):
    vx[i] = np.sqrt(TEMP)*np.random.normal()
    vy[i] = np.sqrt(TEMP)*np.random.normal()

vgx=0.0
vgy=0.0
ave_v=0.0

for i in range(N):
    vgx+=vx[i]
    vgy+=vy[i]

vgx = vgx/N
vgy = vgy/N

for i in range(N):
    vx[i]-=vgx
    vy[i]-=vgy
    ave_v+=np.sqrt(vx[i]**2+vy[i]**2)



#maxV = ave_v/N
    
"""
for i in range(N):
    vx[i]-=vgx/N
    vy[i]-=vgy/N
"""
  
plt.figure(figsize=[lx,ly])

#初期力
PAIRLIST=gmap_create2(N,rx,ry,l_gx,l_gy,n_gx,n_gy,neighbor_row,neighbor_col,neighbor_len,
                 lx,ly,aroundGridNum)
ax,ay,Pot0=force(ax,ay,Pot,N,rx,ry,vx,vy,lx,ly,PAIRLIST)

#rcを定義したい






def Book_Keeping(N,rc, gridWidth, aroundGridNum,vxList,vyList,h):
  maxV=0
  for i in range(N):
      #maxV = max_V
    maxCandidateV=np.sqrt(vxList[i]*vxList[i]+vyList[i]*vyList[i])
    if(maxCandidateV>maxV):
      maxV=maxCandidateV
  
  tLim=(aroundGridNum*gridWidth-rc)/(2*maxV)
  reusableNum=int(tLim/(2*h))
  return reusableNum


for step in range(nsteps):
    print(step)

# =======================================================Snapshot
    l=0
    bx=plt.axes()
    #plt.axis([0,lx,0,ly])
    plt.xlim(0,lx)
    plt.ylim(0,ly)
    
        
    
    
    #グラフのメモリ・ステップ数を消す処理
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    if step%nout==0:
    #bx.set_title(step)
        for i in range(Nx):
            for j in range(Ny):
                
                #粒子に色をつける処理
                if sigma[l] < 0.5:
                    col = 'r'
                elif sigma[l]>=0.5:
                    col = 'g'
                """
                if l==219:
                     col="y"
                """
                     
                
                 
                c=pat.Circle(xy=(rx[l],ry[l]),radius=sigma[l],color=col)
                bx.add_patch(c)
                
                
            #===================================周期境界条件
            #右
                if rx[l] < sigma[l]:
                    c=pat.Circle(xy=(rx[l]+lx,ry[l]),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #左
                elif lx - rx[l] <= sigma[l]:
                    c=pat.Circle(xy=(rx[l]-lx,ry[l]),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #下
                if ry[l] < sigma[l]:
                    c=pat.Circle(xy=(rx[l],ry[l]+ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #上
                elif ly - ry[l] <= sigma[l]:
                    c=pat.Circle(xy=(rx[l],ry[l]-ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #左下
                if rx[l] < sigma[l] and ry[l] <sigma[l]:
                    c=pat.Circle(xy=(rx[l]+lx,ry[l]+ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #右下
                elif lx - rx[l] <= sigma[l] and ry[l]<sigma[l]:
                    c=pat.Circle(xy=(rx[l]-lx,ry[l]+ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #右上
                elif lx - rx[l] <= sigma[l] and ly - ry[l] <= sigma[l]:
                    c=pat.Circle(xy=(rx[l]-lx,ry[l]-ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
            #左上
                elif rx[l] < sigma[l] and ly - ry[l] <= sigma[l]:
                    c=pat.Circle(xy=(rx[l]+lx,ry[l]-ly),radius=sigma[l],color=col)
                    bx.add_patch(c)
                
                
                #粒子番号つける処理
                msg = str(l)
                plt.annotate(msg,xy=(rx[l],ry[l]))

                
                l+=1
                
                
                    
                    
    
        plt.savefig("./figure_binary_pl0.01_0.55/figure{0}.png".format(step))
    plt.cla()

#==================================================MAIN LOOP
#速度ベルレ法

    for i in range(N):
        rx[i]=rx[i]+vx[i]*h+(ax[i]*h*h)/2
        ry[i]=ry[i]+vy[i]*h+(ay[i]*h*h)/2

    ########x:周期境界条件#######
    
        if rx[i]>lx:
            rx[i]-=lx
        elif rx[i]<=0.0:
            rx[i]+=lx
        #else:
            #rx[i]=rx[i]
        
            
    ##########y:周期境界###########
        if ry[i]>ly:
            ry[i]-=ly
        elif ry[i]<=0.0:
            ry[i]+=ly
        #else:
            #ry[i]=ry[i]  
        
        
        vx[i]=vx[i]+ax[i]*h/2
        vy[i]=vy[i]+ay[i]*h/2
    
    #reusableCount:Bookkeepingで計算したグリッドが再利用できる最大数
    if reusableCount==0:
      PAIRLIST=gmap_create2(N,rx,ry,l_gx,l_gy,n_gx,n_gy,neighbor_row,neighbor_col,neighbor_len,lx,ly,aroundGridNum)
      reusableCount=Book_Keeping(N,rc,l_gx_d,aroundGridNum,vx,vy,h)
      #print(pairlist)
    else:
      reusableCount-=1
      
    ax,ay,Pot=force(ax,ay,Pot,N,rx,ry,vx,vy,lx,ly,PAIRLIST)
    noise(ax,ay,N,DL)
    active(ax,ay,N,theta)
    for i in range(N):    
        vx[i]=vx[i]+ax[i]*h/2
        vy[i]=vy[i]+ay[i]*h/2
    
    Kin = 0.0
    for i in range(N):
        Kin+=vx[i]*vx[i]+vy[i]*vy[i]
    Kin = Kin/2
    
    Total_E =Kin+Pot




output_dir="gif"
#############アニメーション################    
images=[]
for i in range(0,nsteps,nout):
    file_name="./figure_binary_pl0.01_0.55/figure"+str(i)+".png"
    im=Image.open(file_name)
    images.append(im)
if not os.path.exists(output_dir): os.makedirs(output_dir)
images[0].save("./gif/binary_pl0.01_0.55.gif",save_all=True,append_images=images[1:]
               ,loop=0,duration=30)
    









    
"""
###############Pot,Total_Eの時間変化プロット########################
x_ax = range(nsteps)
def all_fig():
    plt.plot(x_ax,Pot,label="Pot",color="red")
    plt.plot(x_ax,Total_E,label="Total_E",color="blue")
    
    plt.legend()
    plt.xlabel("t")
    plt.show()

all_fig()



"""






        
        
            

