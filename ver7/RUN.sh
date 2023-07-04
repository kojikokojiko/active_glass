#!/bin/bash



nu_list=(0.72 0.74 0.76 0.78 )
# 0.55 0.6 0.65 0.7 0.75 0.8)
v0_list=(1.0  10.0 50.0 )
fixed_per_list=(0.0)

for nu in "${nu_list[@]}"
do 
    for v0 in  "${v0_list[@]}"
    do
        for fixed_per in  "${fixed_per_list[@]}"
        do
        echo $nu $v0  $fixed_per
        name="${nu}_${v0}_${fixed_per}"
        nohup time python abp_glass.py $nu $v0 $fixed_per> ${name}.log 2>&1 &
        # python animation.py $nu $v0 $fixed_per> ${name}.log 2>&1 &
        done
    done
done

# g++ -std=c++11 -o iabp2 iABP2.cpp
# nohup time ./iabp2 1.0 1.0 10000 # Pe M N
	






# rho=float(sys.argv[1])
# ave_flow=float(sys.argv[2])
# static_dia=float(sys.argv[3])
# reduced_speed=float(sys.argv[5])
# rotational_diffusion=float(sys.argv[6])

