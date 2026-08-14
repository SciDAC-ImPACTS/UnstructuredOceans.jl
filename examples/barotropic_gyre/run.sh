#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J barotropic_gyre
#SBATCH -A m4259
#SBATCH -t 02:00:00

# replace this path with the path to the git repo
gitdir="${HOME}/.julia/dev/UnstructuredOceans.jl"
driver="src/driver/mpas_ocean.jl"
execut="${gitdir}/${driver}"

#module purge
#source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
#module load PrgEnv-gnu/8.5.0
#module load julia/1.10.4
#module load cpe-cuda/23.12

# Unlike the inertial gravity wave, the barotropic gyre is a single forward
# spin-up compared against an analytical streamfunction (see analysis.jl), not
# a multi-resolution convergence study.  The loop is kept so additional
# resolutions can be added trivially.
for dir in 10km; do
    python setup.py --res $(echo $dir | sed 's/[^0-9]//g') --dir $dir

    cd $dir
    cp ../UnstructuredOceans.yaml ./config.yml

    start=$(date +%s.%N)

    #srun -n 1 julia --project=$gitdir -- $execut config.yml
    julia -O0 --color=yes --project=$gitdir -- $execut config.yml $1

    end=$(date +%s.%N)

    runtime=$(echo "$end - $start" | bc)

    echo "${dir} ran in ${runtime} secs"

    cd ..
done
