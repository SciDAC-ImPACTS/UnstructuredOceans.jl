module UnstructuredOceansAMDGPUExt

using UnstructuredOceans
using AMDGPU
using AMDGPU: ROCArray, ROCDeviceArray, ROCBackend
using GPUArraysCore: allowscalar
import UnstructuredOceans: GPU, CPU, architecture, array_type, on_architecture, device!,
             set_ad_device_heap!

function __init__()
    if AMDGPU.has_rocm_gpu()
        @debug "ROCm-enabled GPU(s) detected:"
        for dev in AMDGPU.devices()
            @debug "$dev"
        end
        allowscalar(false)
    end
end

const AMDGPUGPU = GPU{<:ROCBackend}
AMDGPUGPU() = GPU(ROCBackend())

function UnstructuredOceans.GPU()
    AMDGPU.has_rocm_gpu() && return AMDGPUGPU()
    throw(ArgumentError("Cannot make a GPU with the ROCm backend: no ROCm GPU was found!"))
end

Base.summary(::AMDGPUGPU) = "AMDGPUGPU"
UnstructuredOceans.device!(::AMDGPUGPU, i) = AMDGPU.device_id!(i)

UnstructuredOceans.architecture(::ROCArray)            = AMDGPUGPU()
UnstructuredOceans.architecture(::Type{ROCArray})      = AMDGPUGPU()
UnstructuredOceans.architecture(::ROCDeviceArray)      = AMDGPUGPU()
UnstructuredOceans.array_type(::GPU{<:ROCBackend})     = ROCArray

UnstructuredOceans.on_architecture(::AMDGPUGPU, a::Number)       = a
UnstructuredOceans.on_architecture(::CPU, a::ROCArray)           = Array(a)
UnstructuredOceans.on_architecture(::AMDGPUGPU, a::Array)        = ROCArray(a)
UnstructuredOceans.on_architecture(::AMDGPUGPU, a::ROCArray)     = a
UnstructuredOceans.on_architecture(::AMDGPUGPU, a::BitArray)     = ROCArray(a)
UnstructuredOceans.on_architecture(::AMDGPUGPU, a::StepRangeLen) = a

# Raise the device malloc heap for Enzyme reverse-mode AD (see the docstring in
# Architectures.jl). Accept both the UnstructuredOceans GPU arch wrapper and a bare KA
# ROCBackend, since runscripts differentiate against the raw backend. ROCm's
# in-kernel malloc heap is the HIP `hipLimitMallocHeapSize` device limit, set via
# AMDGPU.HIP.heap_size!.
function _set_amdgpu_heap!(bytes::Integer)
    AMDGPU.HIP.heap_size!(bytes)
    return AMDGPU.HIP.heap_size()
end

UnstructuredOceans.set_ad_device_heap!(::AMDGPUGPU; bytes::Integer = 512 * 1024 * 1024) =
    _set_amdgpu_heap!(bytes)
UnstructuredOceans.set_ad_device_heap!(::ROCBackend; bytes::Integer = 512 * 1024 * 1024) =
    _set_amdgpu_heap!(bytes)

end # module UnstructuredOceansAMDGPUExt
