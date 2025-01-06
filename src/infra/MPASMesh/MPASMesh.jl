module MPASMesh

# MPASMesh
export Mesh
# VertMesh.jl
export VerticalMesh
# HorzMesh.jl
export Cell, Edge, Vertex, ReadHorzMesh, HorzMesh

using Accessors
using NCDatasets
using StructArrays
using KernelAbstractions 

import Adapt

const KA = KernelAbstractions

struct Mesh{HM,VM}
    HorzMesh::HM
    VertMesh::VM
end

function Mesh(meshPath::String; nVertLevels=1, backend=KA.CPU())
    # Read in the purely horizontal mesh on the CPU
    HorzMesh = ReadHorzMesh(meshPath; nVertLevels=nVertLevels, backend=KA.CPU())
    # Create a vertical mesh, on the CPU, from the horizontal mesh	
    VertMesh = VerticalMesh(HorzMesh; nVertLevels=nVertLevels, backend=KA.CPU())
    # Create the full Mesh strucutre on the CPU
    MPASMesh = Mesh(HorzMesh, VertMesh)
    # With the full mesh strucutre, now initalize the boundary mask
    setBoundaryMask!(MPASMesh.HorzMesh.Edges, MPASMesh.VertMesh)
    # Adapt the full mesh strcuture to the requested backend
    Adapt.adapt_structure(backend, MPASMesh)
end

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh))
end

include("HorzMesh.jl")
include("VertMesh.jl")

end 
