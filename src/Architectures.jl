import Adapt
import KernelAbstractions as KA

abstract type AbstractArchitecture end
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

# Bridge to the KA.Backend runtime
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

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))

array_type(::CPU) = Array

# Fallback: pass through
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
