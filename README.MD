# `UnstructuredOceans.jl`: a differentiable, performance-portable ocean dynamical core

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://SciDAC-ImPACTS.github.io/UnstructuredOceans.jl/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7493065.svg)](https://doi.org/10.5281/zenodo.7493065)

An ocean model capable of running on irregular, non-rectilinear, TRiSK-based meshes. Inspired by MPAS-Ocean in Fortran.

## Quickstart

```bash
git clone https://github.com/SciDAC-ImPACTS/UnstructuredOceans.jl UnstructuredOceans.jl
cd UnstructuredOceans.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the wind-driven barotropic gyre example (a shipped verification case) on the
CPU, or append `cuda` to run on an NVIDIA GPU:

```bash
julia --project=. src/driver/mpas_ocean.jl examples/barotropic_gyre/10km/config.yml
julia --project=. src/driver/mpas_ocean.jl examples/barotropic_gyre/10km/config.yml cuda
```

## Features

- **Unstructured grids** — TRiSK discretization on MPAS Voronoi meshes.
- **Performance portable** — one KernelAbstractions.jl kernel set runs on CPU and
  on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI) GPUs, selected at runtime via
  package extensions.
- **Differentiable** — forward- and checkpointed reverse-mode gradients via
  Enzyme.jl and Checkpointing.jl, validated against finite differences.
