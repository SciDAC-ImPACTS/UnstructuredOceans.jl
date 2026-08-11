```@meta
CurrentModule = MOKA
```

# Choose a compute backend

MOKA runs the same kernels on the CPU and on NVIDIA, AMD, and Intel GPUs. The
backend is selected at runtime and passed into [`ocn_init`](@ref) (and the array
constructors) as a KernelAbstractions backend. This guide shows how to pick one.

## CPU (no extra packages)

```julia
using MOKA
import KernelAbstractions as KA

backend = KA.CPU()
Setup, Diag, Tend, Prog = ocn_init(config; backend = backend)
```

## GPU

GPU support is provided by package extensions that load automatically the moment
the vendor package is in scope alongside `MOKA`. Add and load the one for your
hardware, then construct a [`GPU`](@ref) architecture and take its
[`device`](@ref):

```julia
using MOKA
using CUDA                      # or: using AMDGPU / using oneAPI

arch    = GPU()                 # vendor extension supplies the default device
backend = device(arch)          # the KernelAbstractions backend to allocate on

Setup, Diag, Tend, Prog = ocn_init(config; backend = backend)
```

| Vendor | Package  | Extension        |
|:-------|:---------|:-----------------|
| NVIDIA | `CUDA`   | `MOKACUDAExt`    |
| AMD    | `AMDGPU` | `MOKAAMDGPUExt`  |
| Intel  | `oneAPI` | `MOKAoneAPIExt`  |

The command-line driver exposes the NVIDIA path directly:

```bash
julia --project=. src/driver/mpas_ocean.jl <config.yml> cuda
```

## Moving data between host and device

Use [`on_architecture`](@ref) to move arrays (or tuples/named tuples of arrays)
onto an architecture, and [`architecture`](@ref) to infer where an array
currently lives. Whole model structs (`Mesh`, [`PrognosticVars`](@ref), …)
support `Adapt.adapt` for the same purpose.

## Tuning kernel launches

Every kernel-launching function accepts an `nthreads` keyword (the
KernelAbstractions workgroup size), defaulting to [`DEFAULT_NTHREADS`](@ref)
(`64`, a warp/wavefront multiple). Override it per call to experiment with
occupancy on a given device.

## Before differentiating on a GPU

Reverse-mode AD stores its tape in the GPU's in-kernel `malloc` heap, and the
default heap (~8 MB on CUDA) overflows at model scale, faulting with an illegal
memory access. Raise it once, after selecting the backend and before
differentiating, with [`set_ad_device_heap!`](@ref):

```julia
set_ad_device_heap!(arch)       # no-op on CPU; enlarges the CUDA device heap
```

See [Automatic differentiation](@ref) for the full AD workflow.
