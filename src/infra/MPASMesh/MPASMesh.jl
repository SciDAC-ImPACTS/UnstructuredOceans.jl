module MPASMesh

# MPASMesh
export Mesh
# VertMesh.jl
export VerticalMesh
# Constants.jl
export Constants
# ForcingVars.jl
export ForcingVars
# HorzMesh.jl
export Cell, Edge, Vertex, read_horz_mesh, HorzMesh

using Accessors
using NCDatasets
using StructArrays
using KernelAbstractions 

import Adapt

const KA = KernelAbstractions

"""
    Mesh(HorzMesh, VertMesh)
    Mesh(HorzMesh, VertMesh, Constants)
    Mesh(HorzMesh, VertMesh, Constants, Forcing)

The full model mesh, pairing a horizontal [`HorzMesh`](@ref) (TRiSK cells, edges,
and dual vertices) with a [`VerticalMesh`](@ref) (levels and resting thicknesses),
the physical [`Constants`](@ref) (gravity, density), and the external
[`ForcingVars`](@ref) (wind stress, …). This is the geometry, parameters, and
forcing passed to the operators, tendencies, and time integrator. `Adapt.adapt`
moves the whole mesh between host and device.

The shorter constructors build standard constants and zeroed forcing on the host
backend, so existing callers (unit tests, operator harnesses) keep working
unchanged.
"""
struct Mesh{HM,VM,CT,FT}
    HorzMesh::HM
    VertMesh::VM
    Constants::CT
    Forcing::FT
    # inner constructor should check meshes are
    # on the same backend
end

# Backward-compatible constructors: default constants + zeroed forcing on CPU.
Mesh(HorzMesh, VertMesh) = Mesh(HorzMesh, VertMesh, Constants())
Mesh(HorzMesh, VertMesh, Constants) =
    Mesh(HorzMesh, VertMesh, Constants, ForcingVars(HorzMesh.Edges.nEdges))

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh),
                Adapt.adapt(backend, x.Constants),
                Adapt.adapt(backend, x.Forcing))
end

include("Constants.jl")
include("ForcingVars.jl")
include("HorzMesh.jl")
include("VertMesh.jl")

end 
