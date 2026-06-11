#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J interial_gravity_wave
#SBATCH -A m4259
#SBATCH -t 02:00:00

# replace this path with the path to the git repo
gitdir="${HOME}/.julia/dev/Moka.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

#module purge
#source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
#module load PrgEnv-gnu/8.5.0
#module load julia/1.10.4
#module load cpe-cuda/23.12

#module load julia/1.10.4

for dir in 200km 100km 50km 25km; do
    python setup.py --res $(echo $dir | sed 's/[^0-9]//g') --outdir $dir

    mkdir $dir
    cd $dir
    cp ../Moka.yaml ./config.yml

    start=$(date +%s.%N)

    #srun -n 1 julia --project=$gitdir -- $execut config.yml  
    julia -O0 --color=yes --project=$gitdir -- $execut config.yml

    end=$(date +%s.%N)
    
    runtime=$(awk -v start=$start -v end=$end 'BEGIN {print end - start}')

    echo "${dir} ran in ${runtime} secs"

    cd ..
done  


