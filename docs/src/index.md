```@meta
CurrentModule = MOKA
```

# MOKA.jl

**MOKA.jl** (*MPAS Ocean using Kernel Abstractions*) is a performance-portable,
differentiable shallow-water ocean dynamical core written in Julia. It
discretizes the shallow-water equations with the **TRiSK** scheme on unstructured
Voronoi (**MPAS**) meshes, following the design of the Fortran
[MPAS-Ocean](https://mpas-dev.github.io/) model, the ocean component of E3SM.

## Why MOKA

- **Performance portable.** Every compute kernel is written once with
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and
  runs unchanged on CPUs and on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI)
  GPUs — selected at runtime via package extensions, with no source edits.
- **Differentiable.** Gradients of a simulation are available through
  [Enzyme.jl](https://enzyme.mit.edu/julia/), in both forward mode and
  checkpointed reverse mode (adjoint) via
  [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl),
  validated against finite differences. This enables adjoint-based sensitivity
  analysis, parameter calibration, and gradient-based optimization.
- **Unstructured grids.** Operating on MPAS Voronoi meshes gives accurate
  topography representation and regional mesh refinement.

Some languages are easy to develop in but slow to run; others are fast but hard
to write. Julia aims to be both, which makes it a compelling choice for a modern,
GPU-accelerated, differentiable climate model.

!!! note "Physics scope"
    The current dynamical core integrates the single-layer shallow-water
    equations with gravity (pressure gradient), Coriolis, Laplacian (del2)
    viscosity, and wind-stress forcing. Nonlinear advection is present in the
    code structure but not yet active.

## Installation

```julia
using Pkg
Pkg.add(url = "https://https://github.com/SciDAC-ImPACTS/Moka.jl")
```

To run on a GPU or to compute gradients, add the relevant trigger package into
the same environment; the corresponding extension then loads automatically:

```julia
Pkg.add("CUDA")                         # NVIDIA GPU (or "AMDGPU" / "oneAPI")
Pkg.add(["Enzyme", "Checkpointing"])    # reverse-mode AD
```

## Where to go next

| If you want to…                                   | Read                                                  |
|:--------------------------------------------------|:------------------------------------------------------|
| Run your first simulation, step by step           | [Getting started](@ref)                               |
| Accomplish a specific task (GPU, AD, config)      | the [How-to guides](@ref "Choose a compute backend")  |
| Understand the equations and numerics             | [Governing equations](@ref), [TRiSK discretization](@ref) |
| Look up a function or type                         | the [API](@ref) reference                             |

## Citing

MOKA.jl is described in *"Portable, High-Performance and Differentiable Ocean
Simulations with Unstructured Grids in Julia"* (Seth et al.). If you use the
package in academic work, please cite that paper and the archived software
release.
