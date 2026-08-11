module LayerThickness

export compute_layer_thickness_tendency!

using UnPack
using KernelAbstractions 
using GPUArraysCore: @allowscalar
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!,
            DEFAULT_NTHREADS

const KA = KernelAbstractions

include("horizontal_advection.jl")

function compute_layer_thickness_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendLayerThickness)
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendLayerThickness, Mesh.HorzMesh.PrimaryCells.nCells, ndrange=Mesh.HorzMesh.PrimaryCells.nCells)

    # compute horizontal advection of layer thickness
    horizontal_advection_tendency!(
        Tend, Prog, Diag, Mesh; nthreads=nthreads)
end

end
