module UnstructuredOceansCUDAExt

using UnstructuredOceans
using CUDA
using CUDA: CuArray, CuDeviceArray, CUDABackend
using GPUArraysCore: allowscalar
import UnstructuredOceans: GPU, CPU, architecture, array_type, on_architecture, device!,
             set_ad_device_heap!

# CUDACore owns the context-limit API (`limit!` / `CU_LIMIT_MALLOC_HEAP_SIZE`)
# and is reached without a direct dependency via the module that defines `limit!`.
const _CUDACore = parentmodule(CUDA.limit!)

function __init__()
    if CUDA.functional()
        @debug "CUDA-enabled GPU(s) detected:"
        for (gpu, dev) in enumerate(CUDA.devices())
            @debug "$dev: $(CUDA.name(dev))"
        end
        allowscalar(false)
    end
end

const CUDAGPU = GPU{<:CUDABackend}
CUDAGPU() = GPU(CUDABackend(always_inline=true))

function UnstructuredOceans.GPU()
    CUDA.has_cuda_gpu() && return CUDAGPU()
    throw(ArgumentError("Cannot make a GPU with the CUDA backend: no CUDA GPU was found!"))
end

Base.summary(::CUDAGPU) = "CUDAGPU"
UnstructuredOceans.device!(::CUDAGPU, i) = CUDA.device!(i)

UnstructuredOceans.architecture(::CuArray)            = CUDAGPU()
UnstructuredOceans.architecture(::Type{CuArray})      = CUDAGPU()
UnstructuredOceans.architecture(::CuDeviceArray)      = CUDAGPU()
UnstructuredOceans.array_type(::GPU{<:CUDABackend})   = CuArray

UnstructuredOceans.on_architecture(::CUDAGPU, a::Number)       = a
UnstructuredOceans.on_architecture(::CPU, a::CuArray)          = Array(a)
UnstructuredOceans.on_architecture(::CUDAGPU, a::Array)        = CuArray(a)
UnstructuredOceans.on_architecture(::CUDAGPU, a::CuArray)      = a
UnstructuredOceans.on_architecture(::CUDAGPU, a::BitArray)     = CuArray(a)
UnstructuredOceans.on_architecture(::CUDAGPU, a::StepRangeLen) = a

# Raise the device malloc heap for Enzyme reverse-mode AD (see the docstring in
# Architectures.jl). Accept both the UnstructuredOceans GPU arch wrapper and a bare KA
# CUDABackend, since runscripts differentiate against the raw backend.
function _set_cuda_heap!(bytes::Integer)
    _CUDACore.limit!(_CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE, bytes)
    return _CUDACore.limit(_CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE)
end

UnstructuredOceans.set_ad_device_heap!(::CUDAGPU; bytes::Integer = 512 * 1024 * 1024) =
    _set_cuda_heap!(bytes)
UnstructuredOceans.set_ad_device_heap!(::CUDABackend; bytes::Integer = 512 * 1024 * 1024) =
    _set_cuda_heap!(bytes)

end # module UnstructuredOceansCUDAExt
