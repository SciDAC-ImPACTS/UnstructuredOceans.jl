module MOKACUDAExt

using MOKA
using CUDA
using CUDA: CuArray, CuDeviceArray, CUDABackend
using GPUArraysCore: allowscalar
import MOKA: GPU, CPU, architecture, array_type, on_architecture, device!

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

end # module MOKACUDAExt
