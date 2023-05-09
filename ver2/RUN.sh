#!/bin/bash



nu_list=(0.55 0.65 0.7  )
# 0.55 0.6 0.65 0.7 0.75 0.8)
pl_list=(1.0  10.0 50.0 200.0)

for nu in "${nu_list[@]}"
do 
    for pl in  "${pl_list[@]}"
    do
        echo $nu $pl 
        name="${nu}_${pl}"
        nohup time python abp_glass.py $nu $pl > ${name}.log 2>&1 &
        # nohup time python animation.py $nu $pl > ${name}.log 2>&1 &
    done
done

# g++ -std=c++11 -o iabp2 iABP2.cpp
# nohup time ./iabp2 1.0 1.0 10000 # Pe M N
	






# rho=float(sys.argv[1])
# ave_flow=float(sys.argv[2])
# static_dia=float(sys.argv[3])
# reduced_speed=float(sys.argv[5])
# rotational_diffusion=float(sys.argv[6])

