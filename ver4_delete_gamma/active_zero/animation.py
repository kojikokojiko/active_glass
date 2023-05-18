import os 
from PIL import Image
import sys
ver=1
nu=float(sys.argv[1])
pl=float(sys.argv[2])
fixed_percent=float(sys.argv[3])

# N=2500

# red_v=format(red_v,'.6f')
# rotate_dif=format(rotate_dif,'.6f')
# rho=format(rho,'.2f')
# va=format(va,'.2f')
ver=str(nu)+"_"+str(pl)+"_"+str(fixed_percent)
main_dir="./"+ver
print(main_dir)

output_dir=main_dir+"/abpgif2"

pic_output=main_dir+"/figure_2d"
############アニメーション################    
images=[]
# image_num=sum(os.path.isfile(os.path.join(pic_output name)) for name in os.listdir(pic_output))
image_num=sum(os.path.isfile(os.path.join(pic_output,name))for name in os.listdir(pic_output))
print(image_num)
for i in range(1,image_num,2):
    file_name=pic_output+"/figure"+str(i)+".png"
    im=Image.open(file_name)
    images.append(im)
if not os.path.exists(output_dir): os.makedirs(output_dir)
images[0].save("./"+output_dir+"/out_ela2.gif",save_all=True,append_images=images[1:],loop=0,duration=10)
    