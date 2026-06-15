#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J interial_gravity_wave
#SBATCH -A m4259
#SBATCH -t 02:00:00

# replace this path with the path to the git repo
gitdir="${HOME}/.julia/dev/MOKA.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

for dir in 200km 100km 50km 25km; do
    python setup.py --res $(echo $dir | sed 's/[^0-9]//g') --dir $dir

    mkdir $dir
    cd $dir
    cp ../Moka.yaml ./config.yml

    # Scale dt linearly with cell width to hold a constant CFL (~0.24), so the
    # RK4 temporal error (~dx^4) stays subdominant to the spatial error (~dx^2)
    # and the spatial convergence order (~2) is measured cleanly.
    case $dir in
        200km) dt="0000_00:04:00" ;;  # 480 s
        100km) dt="0000_00:02:00" ;;  # 240 s
        50km)  dt="0000_00:01:00" ;;  # 120 s
        25km)  dt="0000_00:00:30" ;;  #  60 s
    esac
    sed -i.bak "s/config_dt:.*/config_dt: ${dt}/" config.yml && rm -f config.yml.bak

    start=$(date +%s.%N)

    #srun -n 1 julia --project=$gitdir -- $execut config.yml  
    julia -O0 --color=yes --project=$gitdir -- $execut config.yml

    end=$(date +%s.%N)
    
    runtime=$(echo "$end - $start" | bc)

    echo "${dir} ran in ${runtime} secs"

    cd ..
done  


