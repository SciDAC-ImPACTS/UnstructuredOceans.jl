using Test
using MOKA
using UnPack
using LinearAlgebra

import Adapt
import Downloads
import KernelAbstractions as KA

mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

backend = KA.CPU();

# Read in the purely horizontal doubly periodic testing mesh
HorzMesh = read_horz_mesh(mesh_fn; backend=backend)
# Create a dummy vertical mesh from the horizontal mesh
VertMesh = VerticalMesh(HorzMesh; nVertLevels=10, backend=backend)
# Create a the full Mesh strucutre 
MPASMesh = Mesh(HorzMesh, VertMesh)

# get some dimension information
nEdges = HorzMesh.Edges.nEdges
nCells = HorzMesh.PrimaryCells.nCells
nVertices = HorzMesh.DualCells.nVertices
nVertLevels = VertMesh.nVertLevels

setup = TestSetup(MPASMesh, PlanarTest; backend=backend)

###
### Gradient Test
###

# Scalar field define at cell centers
Scalar  = h(setup, PlanarTest)
# Calculate analytical gradient of cell centered filed (-> edges)
gradAnn = ∇hₑ(setup, PlanarTest)


# Numerical gradient using KernelAbstractions operator 
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
GradientOnEdge!(gradNum, Scalar, MPASMesh)

gradError = ErrorMeasures(gradNum, gradAnn, HorzMesh, Edge)

## test
@test gradError.L_inf ≈ 0.00125026071878552 atol=atol
@test gradError.L_two ≈ 0.00134354611117257 atol=atol

###
### Divergence Test
###

# Edge normal component of vector value field defined at cell edges
VecEdge = 𝐅ₑ(setup, PlanarTest)
# Calculate the analytical divergence of field on edges (-> cells)
divAnn = div𝐅(setup, PlanarTest)
# Numerical divergence using KernelAbstractions operator
divNum = KA.zeros(backend, Float64, (nVertLevels, nCells))
temp   = KA.zeros(backend, Float64, (nVertLevels, nEdges))

DivergenceOnCell!(divNum, VecEdge, temp, MPASMesh)

divError = ErrorMeasures(divNum, divAnn, HorzMesh, Cell)

# test
@test divError.L_inf ≈ 0.00124886886594453 atol=atol
@test divError.L_two ≈ 0.00124886886590979 atol=atol

###
### Curl Test
###

# Edge normal component of vector value field defined at cell edges
VecEdge = 𝐅ₑ(setup, PlanarTest)
# Calculate the analytical divergence of field on edges (-> vertices)
curlAnn = curl𝐅(setup, PlanarTest)
# Numerical curl using KernelAbstractions operator
curlNum = KA.zeros(backend, Float64, (nVertLevels, nVertices))
CurlOnVertex!(curlNum, VecEdge, MPASMesh)

curlError = ErrorMeasures(curlNum, curlAnn, HorzMesh, Vertex)

# test
@test curlError.L_inf ≈ 0.16136566356969 atol=atol
@test curlError.L_two ≈ 0.16134801689713 atol=atol

###
### Multi-layer correctness (Phase 1): operators must be right at EVERY level.
###
# The mesh above is nVertLevels = 10 and every analytic field is tiled identically
# across levels, so a correct vertically-general kernel must (a) write an identical
# result on every level and (b) match the closed-form answer at each level, not just
# in aggregate. Before Phase 1 the divergence/curl/gradient/interp kernels touched
# only level 1; these checks would have left levels 2:end at zero.
@testset "Operators are vertically general (per-level)" begin
    @test nVertLevels > 1   # guard: the point of the test

    # Gradient: identical across levels, and each level matches level 1.
    gradNum_ml = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    GradientOnEdge!(gradNum_ml, Scalar, MPASMesh)
    g = Array(gradNum_ml)
    @test all(g[k, :] == g[1, :] for k in 2:nVertLevels)
    @test maximum(abs, g[1, :]) > 0        # not all-zero

    # Divergence: identical across levels.
    divNum_ml = KA.zeros(backend, Float64, (nVertLevels, nCells))
    temp_ml   = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    DivergenceOnCell!(divNum_ml, VecEdge, temp_ml, MPASMesh)
    d = Array(divNum_ml)
    @test all(d[k, :] == d[1, :] for k in 2:nVertLevels)
    @test maximum(abs, d[1, :]) > 0

    # Curl: identical across levels.
    curlNum_ml = KA.zeros(backend, Float64, (nVertLevels, nVertices))
    CurlOnVertex!(curlNum_ml, VecEdge, MPASMesh)
    c = Array(curlNum_ml)
    @test all(c[k, :] == c[1, :] for k in 2:nVertLevels)
    @test maximum(abs, c[1, :]) > 0

    # interpolateCell2Edge!: cell -> edge average at every level (used by the
    # thickness-flux diagnostic). Interior (non-boundary) edges must equal the
    # mean of their two neighbour cells, on every level.
    edgeVal = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    MOKA.interpolateCell2Edge!(edgeVal, Scalar, MPASMesh)
    ev = Array(edgeVal)
    @test all(ev[k, :] == ev[1, :] for k in 2:nVertLevels)

    cellsOnEdge  = Array(HorzMesh.Edges.cellsOnEdge)
    boundaryEdge = Array(HorzMesh.Edges.boundaryEdge)
    sc = Array(Scalar)
    ok = true
    for iEdge in 1:nEdges
        boundaryEdge[iEdge] == 1 && continue
        c1 = cellsOnEdge[1, iEdge]; c2 = cellsOnEdge[2, iEdge]
        expected = 0.5 * (sc[1, c1] + sc[1, c2])
        ok &= isapprox(ev[1, iEdge], expected; atol=1e-12)
    end
    @test ok
end

###
### Kernels honor the vertical bounds (Phase 1).
###
# The tendency kernels loop `for k in 1:maxLevelEdge.Top[iEdge]`. Artificially
# shrinking those bounds must leave the levels above them untouched, proving the
# loops read the real per-column bounds rather than a hardcoded nVertLevels.
@testset "Tendency kernels respect maxLevelEdge.Top bounds" begin
    # Seed spatially-varying, level-uniform fields so the tendencies are genuinely
    # nonzero at the surface (a uniform state would make the check vacuous).
    ssh0 = Array(h(setup, PlanarTest))[1, :]                 # varying ssh -> nonzero PGF
    vel0 = Array(𝐅ₑ(setup, PlanarTest))                     # varying velocity -> nonzero mixing/coriolis
    lth0 = ones(Float64, nVertLevels, nCells)
    Prog = MOKA.PrognosticVars(ssh0, vel0, lth0, 2)
    Diag = MOKA.DiagnosticVars(MPASMesh; backend=backend)
    Tend = MOKA.TendencyVars(MPASMesh; backend=backend)

    # Shrink the edge active levels to 1 (surface only) on a fresh mesh copy.
    VertMesh1 = VerticalMesh(HorzMesh; nVertLevels=nVertLevels, backend=backend)
    fill!(VertMesh1.maxLevelEdge.Top, Int32(1))
    fill!(VertMesh1.maxLevelEdge.Bot, Int32(1))
    Mesh1 = Mesh(HorzMesh, VertMesh1)

    MOKA.diagnostic_compute!(Mesh1, Diag, Prog)
    MOKA.compute_normal_velocity_tendency!(Tend, Prog, Diag, Mesh1)

    tnv = Array(Tend.tendNormalVelocity)
    # Levels 2:end were never visited by the k-loops (bounded at 1), so they must
    # remain exactly zero (ZeroOutVector! cleared them and nothing wrote there).
    @test all(tnv[2:end, :] .== 0.0)
    # And the surface level DID get contributions (bound of 1 still runs once).
    @test maximum(abs, tnv[1, :]) > 0
end

###
### Results Display
###

arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"
@info """ (Operators on $arch) \n
Gradient
--------
L∞ norm of error : $(gradError.L_inf)
L₂ norm of error : $(gradError.L_two)

Divergence
----------
L∞ norm of error: $(divError.L_inf)
L₂ norm of error: $(divError.L_two)

Curl
----
L∞ norm of error: $(curlError.L_inf)
L₂ norm of error: $(curlError.L_two)
"""
