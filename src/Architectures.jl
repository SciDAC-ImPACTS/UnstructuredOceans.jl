import Adapt
import KernelAbstractions as KA

# Default KernelAbstractions workgroup size (threads per group) for every compute
# kernel launch. 64 is a warp/wavefront multiple (2× NVIDIA's 32, an even divisor of
# AMD's 64), so no lanes sit idle — unlike the historical 50, which left 14 dead lanes
# in the second 32-wide warp. Every kernel-launching function takes an `nthreads`
# keyword defaulting to this, so it can be tuned per call/architecture without editing
# the launch sites. See the benchmark rationale in kernel_benchmark.jl.
"""
    DEFAULT_NTHREADS

Default KernelAbstractions workgroup size (threads per group), `64`, used by every
compute-kernel launch unless overridden via each launcher's `nthreads` keyword.
It is a warp/wavefront multiple (2× NVIDIA's 32, an even divisor of AMD's 64) so
no lanes sit idle.
"""
const DEFAULT_NTHREADS = 64

"""
    AbstractArchitecture

Supertype for the compute architectures MOKA runs on. Concrete subtypes are
[`CPU`](@ref) and [`GPU`](@ref); use [`device`](@ref) to obtain the corresponding
KernelAbstractions backend.
"""
abstract type AbstractArchitecture end

"""
    AbstractSerialArchitecture <: AbstractArchitecture

Supertype for single-node (non-distributed) architectures — the ones currently
supported, [`CPU`](@ref) and [`GPU`](@ref).
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

"""
    CPU <: AbstractSerialArchitecture

Run MOKA on one CPU node.
"""
struct CPU <: AbstractSerialArchitecture end

"""
    GPU(device)

Return a GPU architecture using `device`.
`device` defaults to `CUDA.CUDABackend(always_inline=true)` if CUDA is loaded.
"""
struct GPU{D} <: AbstractSerialArchitecture
    device :: D
end

"""
    device(arch::AbstractArchitecture)

Return the KernelAbstractions backend for `arch`: `KernelAbstractions.CPU()` for
[`CPU`](@ref), or the stored device backend for a [`GPU`](@ref). This backend is
what gets passed as the `backend` keyword to [`ocn_init`](@ref) and the array
constructors.
"""
device(::CPU) = KA.CPU()
device(a::GPU) = a.device

device!(::CPU, i) = nothing

"""
    set_ad_device_heap!(arch; bytes)

Raise the device-side dynamic-allocation (`malloc`) heap so that Enzyme
reverse-mode AD has room for its tape.

Enzyme's split reverse mode stores the per-thread tape in buffers obtained from
the GPU's in-kernel `malloc`. The default device heap (~8 MB on CUDA) overflows
once the differentiated kernels run at model scale (e.g. the coriolis tendency
scatters over `nEdges × nEdgesOnEdge` entries); `malloc` then returns NULL and
the augmented-forward kernel faults with an illegal memory access. Call this
once, after selecting the backend and before `autodiff`. No-op on CPU and on
backends that don't need it; the CUDA implementation lives in `MOKACUDAExt`.

Returns the heap size in bytes that is in effect afterwards (or `nothing`).
"""
set_ad_device_heap!(arch; bytes::Integer = 512 * 1024 * 1024) = nothing
set_ad_device_heap!(::CPU; bytes::Integer = 0) = nothing

synchronize(::CPU) = KA.synchronize(KA.CPU())
synchronize(a::AbstractArchitecture) = KA.synchronize(a.device)

"""
    architecture(x)

Infer the [`AbstractArchitecture`](@ref) that an array (or array-like) `x` lives
on. Returns [`CPU`](@ref)`()` for a host `Array`, and `nothing` for scalars.
GPU array types register their own methods in the vendor extensions.
"""
architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))

"""
    array_type(arch::AbstractArchitecture)

Return the array type used on `arch` (`Array` for [`CPU`](@ref); the device array
type for a [`GPU`](@ref), registered by the vendor extension).
"""
array_type(::CPU) = Array

"""
    on_architecture(arch, x)

Move `x` (an array, tuple, or named tuple of arrays) onto architecture `arch`,
returning host arrays for [`CPU`](@ref) and device arrays for a [`GPU`](@ref).
Values already on the target (or that need no transfer, e.g. ranges) are passed
through unchanged.
"""
on_architecture(arch, a) = a

# Back-compat: raw KA.Backend form used in test/ocn/test_GPU.jl
on_architecture(backend::KA.Backend, array::AbstractArray) = Adapt.adapt_storage(backend, array)

# CPU-specific overrides
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::BitArray) = a
on_architecture(::CPU, a::StepRangeLen) = a

on_architecture(arch::AbstractSerialArchitecture, t::Tuple) =
    Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) =
    NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

Base.summary(::CPU) = "CPU"
Base.summary(gpu::GPU) = "GPU{$(typeof(gpu.device))}"

###
### Helper functions for constructing PrognosticVars, DiagnosticVars,
### TendencyVars, and ForcingVars structures
###

function check_typeof_args(args::Tuple)
    if !allequal(nameof.(typeof.(args)))
        error("Input arguments must be of all the same type")
    end
end

function check_args_backend(args::Tuple)
    if !allequal(KA.get_backend.(args))
        error("All input arguments must have the same backend")
    end
end

function check_eltype_args(args::Tuple)
    if !allequal(eltype.(args))
        error("All input arguments must have the same eltype")
    end
    type, = eltype.(args)
    return type
end
