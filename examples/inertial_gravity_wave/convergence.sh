#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J interial_gravity_wave
#SBATCH -A m4259
#SBATCH -t 02:00:00

# replace this path with the path to the git repo
gitdir="/disk/aseth/UnstructuredOceans.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

for dir in 200km 100km 50km 25km; do
    python setup.py --res $(echo $dir | sed 's/[^0-9]//g') --dir $dir

    cd $dir
    cp ../UnstructuredOceans.yaml ./config.yml

    # Scale dt linearly with cell width at constant CFL ≈ 0.24 (c=√(gH)≈99 m/s).
    # RK4's temporal error is 4th order only asymptotically; at larger CFL it is
    # pre-asymptotic and pollutes the O(Δx²) spatial measurement, so keep CFL
    # modest. This is still ~10x cheaper at 25 km than the old quadratic dt∝Δx²
    # (600 steps vs 6000). dt must divide the 10 h run duration (36000 s).
    case $dir in
        200km) dt="0000_00:08:00" ;;  # 480 s  (CFL 0.24)
        100km) dt="0000_00:04:00" ;;  # 240 s  (CFL 0.24)
        50km)  dt="0000_00:02:00" ;;  # 120 s  (CFL 0.24)
        25km)  dt="0000_00:01:00" ;;  #  60 s  (CFL 0.24)
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


