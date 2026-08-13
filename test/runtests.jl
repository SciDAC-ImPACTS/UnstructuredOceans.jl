using Test
using UnstructuredOceans

include("utilities.jl")

# GPU tests require the UnstructuredOceansCUDAExt extension to be loaded (CUDA.jl installed)
# and actual GPU hardware available (GPU() succeeds).
const _cuda_ext_loaded = !isnothing(Base.get_extension(UnstructuredOceans, :UnstructuredOceansCUDAExt))
const _has_gpu = _cuda_ext_loaded && try (GPU(); true) catch; false end

@testset "UnstructuredOceans" begin

    @testset "Infrastructure Tests" begin
        include("infra/test_Config.jl")
        include("infra/test_timeManager.jl")
    end

    @testset "Operator/Kernel Tests" begin
        include("ocn/test_Operators.jl")
    end

    if _has_gpu
        @testset "GPU Tests" begin
            include("ocn/test_GPU.jl")
        end
        @testset "Enzyme Tests" begin
            include("enzyme/test_Enzyme_Operators.jl")
        end
    end
end
