#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J barotropic_gyre
#SBATCH -A m4259
#SBATCH -t 04:00:00

# replace this path with the path to the git repo
gitdir="/disk/aseth/UnstructuredOceans.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

# Spatial convergence study for the Munk barotropic gyre.  Each resolution is spun
# up to the same 20-day quasi-steady state and compared against the analytical Munk
# streamfunction by convergence.jl.  Only resolutions that resolve
# the ~58 km Munk boundary layer (>= ~3 cells) are in the asymptotic regime, so the
# sweep stays at 20 km and finer (see the warning printed by setup.py).
for dir in 40km 20km 10km 5km 2.5km; do
    python setup.py --res $(echo $dir | sed 's/[^0-9]//g') --dir $dir

    cd $dir
    cp ../UnstructuredOceans.yaml ./config.yml

    # Scale dt linearly with cell width to hold the surface-gravity-wave CFL
    # constant (c = sqrt(gH) ~ 221 m/s; at 10 km / 40 s the CFL is ~0.89).  Each dt
    # divides the 20-day (1 728 000 s) run duration exactly.
    case $dir in
        40km) dt="0000_00:02:40" ;;  # 160 s (CFL 0.89)
        20km) dt="0000_00:01:20" ;;  # 80 s  (CFL 0.89)
        10km) dt="0000_00:00:40" ;;  # 40 s  (CFL 0.89)
        5km)  dt="0000_00:00:20" ;;  # 20 s  (CFL 0.89)
        2.5km) dt="0000_00:00:10" ;;  # 10 s  (CFL 0.89)
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
