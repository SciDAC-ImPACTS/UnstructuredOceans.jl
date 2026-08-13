module UnstructuredOceansoneAPIExt

using UnstructuredOceans
using oneAPI
using oneAPI: oneArray, oneDeviceArray, oneAPIBackend
using GPUArraysCore: allowscalar
import UnstructuredOceans: GPU, CPU, architecture, array_type, on_architecture, device!,
             set_ad_device_heap!

function __init__()
    if oneAPI.functional()
        @debug "oneAPI-enabled GPU(s) detected:"
        for dev in oneAPI.devices()
            @debug "$dev"
        end
        allowscalar(false)
    end
end

const oneAPIGPU = GPU{<:oneAPIBackend}
oneAPIGPU() = GPU(oneAPIBackend())

function UnstructuredOceans.GPU()
    oneAPI.functional() && return oneAPIGPU()
    throw(ArgumentError("Cannot make a GPU with the oneAPI backend: no oneAPI GPU was found!"))
end

Base.summary(::oneAPIGPU) = "oneAPIGPU"
UnstructuredOceans.device!(::oneAPIGPU, i) = oneAPI.device!(i)

UnstructuredOceans.architecture(::oneArray)            = oneAPIGPU()
UnstructuredOceans.architecture(::Type{oneArray})      = oneAPIGPU()
UnstructuredOceans.architecture(::oneDeviceArray)      = oneAPIGPU()
UnstructuredOceans.array_type(::GPU{<:oneAPIBackend})  = oneArray

UnstructuredOceans.on_architecture(::oneAPIGPU, a::Number)       = a
UnstructuredOceans.on_architecture(::CPU, a::oneArray)           = Array(a)
UnstructuredOceans.on_architecture(::oneAPIGPU, a::Array)        = oneArray(a)
UnstructuredOceans.on_architecture(::oneAPIGPU, a::oneArray)     = a
UnstructuredOceans.on_architecture(::oneAPIGPU, a::BitArray)     = oneArray(a)
UnstructuredOceans.on_architecture(::oneAPIGPU, a::StepRangeLen) = a

# Raise the device malloc heap for Enzyme reverse-mode AD (see the docstring in
# Architectures.jl). Accept both the UnstructuredOceans GPU arch wrapper and a bare KA
# oneAPIBackend, since runscripts differentiate against the raw backend. Intel's
# Level Zero / SYCL runtime does not expose a user-tunable in-kernel malloc heap
# limit (unlike CUDA's CU_LIMIT_MALLOC_HEAP_SIZE or ROCm's hipLimitMallocHeapSize),
# so this is a no-op that simply reports the request as satisfied.
UnstructuredOceans.set_ad_device_heap!(::oneAPIGPU; bytes::Integer = 512 * 1024 * 1024) = nothing
UnstructuredOceans.set_ad_device_heap!(::oneAPIBackend; bytes::Integer = 512 * 1024 * 1024) = nothing

end # module UnstructuredOceansoneAPIExt
