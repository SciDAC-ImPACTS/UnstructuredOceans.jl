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
