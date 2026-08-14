module LayerThickness

export compute_layer_thickness_tendency!

using UnPack
using KernelAbstractions 
using GPUArraysCore: @allowscalar
using UnstructuredOceans: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!,
            DEFAULT_NTHREADS

const KA = KernelAbstractions

include("horizontal_advection.jl")

function compute_layer_thickness_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendLayerThickness)
    nCells = Mesh.HorzMesh.PrimaryCells.nCells
    nVertLevels = Mesh.VertMesh.nVertLevels
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendLayerThickness, nCells, ndrange=(nCells, nVertLevels))

    # compute horizontal advection of layer thickness
    horizontal_advection_tendency!(
        Tend, Prog, Diag, Mesh; nthreads=nthreads)
end

end
