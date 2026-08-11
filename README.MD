# :coffee: `Moka.jl`: MPAS Ocean (using) kernel abstractions 

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://GodotMisogi.github.io/Moka.jl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7493065.svg)](https://doi.org/10.5281/zenodo.7493065)

An ocean model capable of running on irregular, non-rectilinear, TRiSK-based meshes. Inspired by MPAS-Ocean in Fortran.

**Why remake a great, working ocean model in Julia?**

Some languages are easy to develop at the cost of executing slowly, while others are lightning fast at run time but much more difficult to write. Julia is a programming language that aims to be the best of both worlds, a development and production language at the same time. To test Julia’s utility in scientific high-performance computing (HPC), we built a MPAS Shallow Water core in Julia and compared it with existing codes. We began with a simple single-processor implementation of the equations, and then ran unit tests and convergence studies for verification purpose. Subsequently, we parallelized the code by rewriting the time integrators as graphics card kernels. The GPU accelerated model achieved an amazing 500 times performance boost from the single-processor version. Finally, to make our model comparable to Fortran MPAS-Ocean, we wrote methods to divide the computational work over multiple cores of a cluster with MPI. We then performed equivalent simulations with the Julia and Fortran codes to compare the speeds, and learn how useful Julia might be for climate modeling and other HPC applications.


Currently only includes gravity, coriolis terms (no non-linear terms).



## Documentation

Full documentation — installation, a step-by-step tutorial, how-to guides
(GPU backends, automatic differentiation, configuration, benchmarks), the
governing equations and TRiSK discretization, and the API reference — is at:

**https://GodotMisogi.github.io/Moka.jl/**

## Quickstart

```bash
git clone https://https://github.com/SciDAC-ImPACTS/Moka.jl MOKA.jl
cd MOKA.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the wind-driven barotropic gyre example (a shipped verification case) on the
CPU, or append `cuda` to run on an NVIDIA GPU:

```bash
julia --project=. src/driver/mpas_ocean.jl examples/barotropic_gyre/10km/config.yml
julia --project=. src/driver/mpas_ocean.jl examples/barotropic_gyre/10km/config.yml cuda
```

See the [Getting started](https://GodotMisogi.github.io/Moka.jl/tutorials/getting_started/)
tutorial for the equivalent Julia-API workflow.

## Features

- **Performance portable** — one KernelAbstractions.jl kernel set runs on CPU and
  on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI) GPUs, selected at runtime via
  package extensions.
- **Differentiable** — forward- and checkpointed reverse-mode gradients via
  Enzyme.jl and Checkpointing.jl, validated against finite differences.
- **Unstructured grids** — TRiSK discretization on MPAS Voronoi meshes.

Verification cases (`examples/`): a wind-driven barotropic (Munk) gyre and an
inertial gravity wave, each with an analytic reference, plus AD validation tests
and performance benchmarks.
