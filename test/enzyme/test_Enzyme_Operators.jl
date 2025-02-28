using Test
using CUDA
using MOKA
using UnPack
using CUDA: @allowscalar
using Enzyme

# Setting meshes to inactive types:
#Enzyme.EnzymeRules.inactive_type(::Type{<:HorzMesh}) = true

import Adapt
import Downloads
import KernelAbstractions as KA

mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

let 
backends = [KA.CPU(), CUDABackend()]
for backend in backends
    @show backend

    # Create a the full Mesh strucutre 
    MPASMesh = Mesh(mesh_fn; nVertLevels=1, backend=backend)
    # Unpack the horizontal and vertcial mesh structs
    @unpack HorzMesh, VertMesh = MPASMesh

    setup = TestSetup(MPASMesh, PlanarTest; backend=backend)

    nEdges = HorzMesh.Edges.nEdges
    nCells = HorzMesh.PrimaryCells.nCells
    nVertLevels = VertMesh.nVertLevels

    ###
    ### Here, we will test Enzyme AD on our kernels
    ###

    # As a clean / easy to read test, let's create an outer function that measures the squared norm of the gradient computed by kernel:
    function gradient_test(grad, hᵢ, mesh::Mesh, backend)
        GradientOnEdge!(grad, hᵢ, mesh::Mesh; backend=backend)
    end

    # Let's recreate all the variables:
    gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    Scalar  = h(setup, PlanarTest)

    d_gradNum  = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    d_Scalar   = KA.zeros(backend, eltype(setup.xᶜ), (nVertLevels, nCells))
    d_MPASMesh = Enzyme.make_zero(MPASMesh)

    kBegin = 1
    kEnd = 1
    @allowscalar d_gradNum[kEnd] = 1.0

    d_normSq = autodiff(Enzyme.Reverse,
                        gradient_test,
                        Duplicated(gradNum, d_gradNum),
                        Duplicated(Scalar, d_Scalar),
                        Duplicated(MPASMesh, d_MPASMesh),
                        Const(backend))
    @allowscalar dnorm_dscalar_rev = d_Scalar[kBegin]

    # Create a the full Mesh strucutre 
    MPASMesh = Mesh(mesh_fn; nVertLevels=1, backend=backend)
    # Unpack the horizontal and vertcial mesh structs
    @unpack HorzMesh, VertMesh = MPASMesh

    setup = TestSetup(MPASMesh, PlanarTest; backend=backend)
    fill!(d_gradNum, 0.0)
    fill!(d_Scalar, 0.0)
    d_MPASMesh = Enzyme.make_zero(MPASMesh)
    @allowscalar d_Scalar[kBegin] = 1.0
    fwd_mode = true
    # Forward mode on CUDA fails with Enzyme crash
    try
        d_normSq = autodiff(Enzyme.Forward,
                            gradient_test,
                            Duplicated(gradNum, d_gradNum),
                            Duplicated(Scalar, d_Scalar),
                            Duplicated(MPASMesh, d_MPASMesh),
                            Const(backend))
    catch
        fwd_mode = false
    end
    
    if backend == CUDABackend()
        @test fwd_mode
    else 
        @test fwd_mode
    end

    @allowscalar dnorm_dscalar_fwd = d_gradNum[kEnd]
    MPASMeshFD = Mesh(mesh_fn; nVertLevels=1, backend=backend)
    HorzMeshFD = MPASMeshFD.HorzMesh
    ϵ = 1e-8

    # For comparison, let's compute the derivative by hand for a given scalar entry:
    gradNumFD = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    ScalarFD  = h(setup, PlanarTest)
    ScalarP = deepcopy(ScalarFD)
    ScalarM = deepcopy(ScalarFD)
    @allowscalar ScalarP[kBegin] = ScalarP[kBegin] + abs(ScalarP[kBegin]) * ϵ
    @allowscalar ScalarM[kBegin] = ScalarM[kBegin] - abs(ScalarM[kBegin]) * ϵ

    gradient_test(gradNumFD, ScalarP, MPASMeshFD, backend)
    @allowscalar normP = gradNumFD[kEnd]
    gradNumFD = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    gradient_test(gradNumFD, ScalarM, MPASMeshFD, backend)
    @allowscalar normM = gradNumFD[kEnd]

    @allowscalar dnorm_dscalar_fd = (normP - normM) / (ScalarP[kBegin] - ScalarM[kBegin])

    #@allowscalar @show normP, normM, ScalarP[k], ScalarM[k]

    @info """ (gradients)\n
    For edge global input $kBegin, output $kEnd
    Enzyme computed $dnorm_dscalar_rev
    Finite differences computed $dnorm_dscalar_fd
    """
    @test isapprox(dnorm_dscalar_rev, dnorm_dscalar_fd, atol=1e-6)
    if backend == CUDABackend()
        @test_broken isapprox(dnorm_dscalar_fwd, dnorm_dscalar_fd, atol=1e-6)
    else
        @test isapprox(dnorm_dscalar_fwd, dnorm_dscalar_fd, atol=1e-6)
    end

    ###
    ### Now let's test divergence:
    ###
    function divergence_test(div, 𝐅ₑ, temp, mesh::Mesh, backend)
        DivergenceOnCell!(div, 𝐅ₑ, temp, mesh::Mesh; backend=backend, nthreads=64)
    end

    @show nEdges, nCells

    divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
    VecEdge = 𝐅ₑ(setup, PlanarTest)
    temp    = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))

    d_divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
    d_VecEdge = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
    d_temp    = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))

    kBegin = 2
    kEnd = 1
    @allowscalar d_divNum[kEnd] = 1.0

    d_normSq = autodiff(Enzyme.Reverse,
                        divergence_test,
                        Duplicated(deepcopy(divNum), d_divNum),
                        Duplicated(deepcopy(VecEdge), d_VecEdge),
                        Duplicated(deepcopy(temp), d_temp),
                        Duplicated(deepcopy(MPASMesh), d_MPASMesh),
                        Const(backend))
    @allowscalar dnorm_dvecedge_rev    = d_VecEdge[kBegin]
    # Create a the full Mesh strucutre 
    MPASMesh = Mesh(mesh_fn; nVertLevels=1, backend=backend)
    # Unpack the horizontal and vertcial mesh structs
    @unpack HorzMesh, VertMesh = MPASMesh

    setup = TestSetup(MPASMesh, PlanarTest; backend=backend)
    fill!(d_divNum, 0.0)
    fill!(d_VecEdge, 0.0)
    fill!(d_temp, 0.0)
    d_MPASMesh = Enzyme.make_zero(MPASMesh)

    @allowscalar d_VecEdge[kBegin] = 1.0
    # Forward mode fails on CPU and CUDA with "Enzyme: unhandled forward for jl_f__svec_ref"
    fwd_mode = false
    try
        d_normSq = autodiff(Enzyme.Forward,
                            divergence_test,
                            Duplicated(divNum, d_divNum),
                            Duplicated(VecEdge, d_VecEdge),
                            Duplicated(temp, d_temp),
                            Duplicated(MPASMesh, d_MPASMesh),
                            Const(backend))
        fwd_mode = true
    catch e
        fwd_mode = false
    end
    @test fwd_mode

    @allowscalar dnorm_dvecedge_fwd = d_divNum[kEnd]
    MPASMeshFD = Mesh(mesh_fn; nVertLevels=1, backend=backend)
    HorzMeshFD = MPASMeshFD.HorzMesh
    ϵ = 1e-8
    # For comparison, let's compute the derivative by hand for a given VecEdge entry:
    VecEdgeP = 𝐅ₑ(setup, PlanarTest)
    VecEdgeM = 𝐅ₑ(setup, PlanarTest)

    @allowscalar VecEdgeP[kBegin] = VecEdgeP[kBegin] + abs(VecEdgeP[kBegin]) * ϵ
    @allowscalar VecEdgeM[kBegin] = VecEdgeM[kBegin] - abs(VecEdgeM[kBegin]) * ϵ

    divNumFD = KA.zeros(backend, Float64, (nVertLevels, nCells))
    tempFD   = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
    divergence_test(divNumFD, VecEdgeP, tempFD, MPASMeshFD, backend)
    @allowscalar testP = divNumFD[kEnd]

    divNumFD = KA.zeros(backend, Float64, (nVertLevels, nCells))
    tempFD   = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
    divergence_test(divNumFD, VecEdgeM, tempFD, MPASMeshFD, backend)
    @allowscalar testM = divNumFD[kEnd]

    @allowscalar dnorm_dvecedge_fd = (testP - testM) / (VecEdgeP[kBegin] - VecEdgeM[kBegin])

    @info """ (divergence)\n
    For cell global input $kBegin, output $kEnd
    Enzyme computed $dnorm_dvecedge_rev
    Finite differences computed $dnorm_dvecedge_fd
    """
    @test isapprox(dnorm_dvecedge_rev, dnorm_dvecedge_fd, atol=1e-6)
    if backend == KA.CPU()
        @test isapprox(dnorm_dvecedge_fwd, dnorm_dvecedge_fd, atol=1e-6)
    elseif backend == CUDABackend()
        @test isapprox(dnorm_dvecedge_fwd, dnorm_dvecedge_fd, atol=1e-6)
    end
end

end
