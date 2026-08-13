```@meta
CurrentModule = UnstructuredOceans
```

# Software architecture

This page explains how UnstructuredOceans is put together — the module layout, the
performance-portability strategy, and the constraints that keep the model
differentiable on GPUs.

## Module layout

The top-level `UnstructuredOceans` module (`src/UnstructuredOceans.jl`) includes its components in dependency
order:

- **`Architectures.jl`** — the [`CPU`](@ref)/[`GPU`](@ref) types, backend
  bridging ([`device`](@ref), [`on_architecture`](@ref)), and AD heap setup
  ([`set_ad_device_heap!`](@ref)).
- **`infra/`** — configuration parsing (`Config.jl`), the ESMF-style
  [`Clock`](@ref) and alarms (`TimeManager.jl`), the `MPASMesh` submodule
  ([`Mesh`](@ref), [`HorzMesh`](@ref), [`VerticalMesh`](@ref)), and NetCDF I/O
  (`OutPut.jl`).
- **`ocn/`** — the differential [operators](@ref "TRiSK discretization")
  (`Operators.jl`), the prognostic and diagnostic state
  ([`PrognosticVars`](@ref), [`DiagnosticVars`](@ref)), and the tendency
  submodules under `Tendencies/`.
- **`forward/`** — model setup ([`ocn_init`](@ref)), the time integrators
  (`time_integration.jl`), and the run loop and AD drivers (`run_loop.jl`).

State is held in mutable structs of arrays; the geometry lives in an immutable
[`Mesh`](@ref) that is passed as a read-only argument to every kernel.

## Performance portability with KernelAbstractions

Every compute step is a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
`@kernel`, written once and launched on whatever backend the arrays live on.
Switching hardware means loading a different vendor package
([Choose a compute backend](@ref)) — no source changes. Host↔device transfer is
handled uniformly through `Adapt.adapt` and [`on_architecture`](@ref). GPU support
is delivered as package **extensions** (`UnstructuredOceansCUDAExt`, `UnstructuredOceansAMDGPUExt`,
`UnstructuredOceansoneAPIExt`) declared under `[weakdeps]`/`[extensions]` in `Project.toml`,
so the base package has no hard GPU dependency and the extensions activate only
when their trigger package is loaded.

## Differentiability constraints

Making the same kernels differentiable by [Enzyme.jl](https://enzyme.mit.edu/julia/)
on the GPU imposes design rules that explain several otherwise-surprising choices
in the code:

- **No active scalar kernel arguments.** Enzyme's KernelAbstractions reverse rule
  rejects by-value `Float64` kernel arguments ("Active kernel arguments not
  supported on GPU"). Runtime scalars like the timestep are therefore passed as
  **length-1 device arrays** and read on-device (`dt[1]`), and compile-time
  coefficients (the RK4 stage fractions) are carried as `Val` type parameters.
- **No host staging copies inside the differentiated region.** Generic
  `copyto!`/`Base.copy` between host and device lowers to a memcpy carrying a
  gc-transition operand bundle Enzyme cannot differentiate, so the integrator
  fills accumulators with explicit device kernels instead.
- **Structured branches, not `continue`.** Boundary/missing-edge handling uses
  `if iEdge != 0 … end` rather than `continue`; this both differentiates
  correctly and avoids out-of-bounds index-0 reads that ROCm/HIP traps as illegal
  addresses.
- **A raised device malloc heap.** Reverse-mode stores its tape in the GPU's
  in-kernel `malloc` heap, which must be enlarged with
  [`set_ad_device_heap!`](@ref) before differentiating at model scale.

Reverse-mode over a time-dependent run is made tractable by
[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl):
[`ocn_run_loop_checkpointed!`](@ref) differentiates a single [`ocn_step!`](@ref)
of an [`OceanModel`](@ref) at a time under the Revolve scheme, freeing each tape
before the next.

## Extending the model

New physics is added as a tendency kernel under `src/ocn/Tendencies/` and summed
into the appropriate `compute…Tendency!`; because it is a KernelAbstractions
kernel it is portable and — if it respects the constraints above —
differentiable, with no per-architecture or per-gradient code to write by hand.
