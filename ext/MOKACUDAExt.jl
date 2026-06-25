module MOKACUDAExt

using MOKA
using CUDA
using CUDA: CuArray, CuDeviceArray, CUDABackend
using GPUArraysCore: allowscalar
import MOKA: GPU, CPU, architecture, array_type, on_architecture, device!,
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

function MOKA.GPU()
    CUDA.has_cuda_gpu() && return CUDAGPU()
    throw(ArgumentError("Cannot make a GPU with the CUDA backend: no CUDA GPU was found!"))
end

Base.summary(::CUDAGPU) = "CUDAGPU"
MOKA.device!(::CUDAGPU, i) = CUDA.device!(i)

MOKA.architecture(::CuArray)            = CUDAGPU()
MOKA.architecture(::Type{CuArray})      = CUDAGPU()
MOKA.architecture(::CuDeviceArray)      = CUDAGPU()
MOKA.array_type(::GPU{<:CUDABackend})   = CuArray

MOKA.on_architecture(::CUDAGPU, a::Number)       = a
MOKA.on_architecture(::CPU, a::CuArray)          = Array(a)
MOKA.on_architecture(::CUDAGPU, a::Array)        = CuArray(a)
MOKA.on_architecture(::CUDAGPU, a::CuArray)      = a
MOKA.on_architecture(::CUDAGPU, a::BitArray)     = CuArray(a)
MOKA.on_architecture(::CUDAGPU, a::StepRangeLen) = a

# Raise the device malloc heap for Enzyme reverse-mode AD (see the docstring in
# Architectures.jl). Accept both the MOKA GPU arch wrapper and a bare KA
# CUDABackend, since runscripts differentiate against the raw backend.
function _set_cuda_heap!(bytes::Integer)
    _CUDACore.limit!(_CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE, bytes)
    return _CUDACore.limit(_CUDACore.CU_LIMIT_MALLOC_HEAP_SIZE)
end

MOKA.set_ad_device_heap!(::CUDAGPU; bytes::Integer = 512 * 1024 * 1024) =
    _set_cuda_heap!(bytes)
MOKA.set_ad_device_heap!(::CUDABackend; bytes::Integer = 512 * 1024 * 1024) =
    _set_cuda_heap!(bytes)

end # module MOKACUDAExt
