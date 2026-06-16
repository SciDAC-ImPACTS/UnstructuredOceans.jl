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

    # Scale dt quadratically with cell width (dt ∝ Δx²) so temporal error stays
    # O(Δx²) and does not pollute the spatial convergence measurement.
    case $dir in
        200km) dt="0000_00:06:40" ;;  # 400 s  (floor(200e3^2 / 1e8))
        100km) dt="0000_00:01:40" ;;  # 100 s  (floor(100e3^2 / 1e8))
        50km)  dt="0000_00:00:25" ;;  #  25 s  (floor( 50e3^2 / 1e8))
        25km)  dt="0000_00:00:06" ;;  #   6 s  (floor( 25e3^2 / 1e8))
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


