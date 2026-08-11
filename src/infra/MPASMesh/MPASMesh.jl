module MPASMesh

# MPASMesh
export Mesh
# VertMesh.jl
export VerticalMesh
# Constants.jl
export Constants
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

The full model mesh, pairing a horizontal [`HorzMesh`](@ref) (TRiSK cells, edges,
and dual vertices) with a [`VerticalMesh`](@ref) (levels and resting thicknesses)
and the physical [`Constants`](@ref) (gravity, density). This is the geometry and
parameters passed to the operators, tendencies, and time integrator. `Adapt.adapt`
moves the whole mesh between host and device.

The two-argument constructor builds standard constants on the host backend, so
existing callers (unit tests, operator harnesses) keep working unchanged.
"""
struct Mesh{HM,VM,CT}
    HorzMesh::HM
    VertMesh::VM
    Constants::CT
    # inner constructor should check meshes are
    # on the same backend
end

# Backward-compatible constructor: default constants on the CPU backend.
Mesh(HorzMesh, VertMesh) = Mesh(HorzMesh, VertMesh, Constants())

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh),
                Adapt.adapt(backend, x.Constants))
end

include("Constants.jl")
include("HorzMesh.jl")
include("VertMesh.jl")

end 
