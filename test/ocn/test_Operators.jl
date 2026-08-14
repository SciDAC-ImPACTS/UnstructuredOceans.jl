using Test
using MOKA
using UnPack
using LinearAlgebra
using GPUArraysCore: @allowscalar

import Adapt
import Downloads
import KernelAbstractions as KA

mesh_fn = DownloadMesh(PlanarTest)

backend = KA.CPU();

# Read in the purely horizontal doubly periodic testing mesh
HorzMesh = ReadHorzMesh(mesh_fn; backend=backend)
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
@allowscalar GradientOnEdge!(gradNum, Scalar, MPASMesh)

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
@allowscalar CurlOnVertex!(curlNum, VecEdge, MPASMesh)

curlError = ErrorMeasures(curlNum, curlAnn, HorzMesh, Vertex)

# test
@test curlError.L_inf ≈ 0.16136566356969 atol=atol
@test curlError.L_two ≈ 0.16134801689713 atol=atol

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
