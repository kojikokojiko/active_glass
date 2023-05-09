import numpy as np
import sys
import os


nu=float(sys.argv[1])



file_name="nu"+str(nu)+".npz"


npz=np.load(file_name)

print(npz.files)

rx=npz["rx"]
ry=npz["ry"]
sigma=npz["sigma"]
typeid=npz["typeid"]


print(rx)
print(sigma)

output_dir="./nu"+str(nu)
os.makedirs(output_dir)
print(output_dir)
os.chdir(output_dir)
np.savetxt("rx.txt",rx)
np.savetxt("ry.txt",ry)
np.savetxt("sigma.txt",sigma)
np.savetxt("typeid.txt",typeid)
np.save("rx.npy",rx)
np.save("ry.npy",ry)
np.save("sigma.npy",sigma)
np.save("typeid.npy",typeid)

