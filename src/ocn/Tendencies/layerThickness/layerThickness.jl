module layerThickness

export computeLayerThicknessTendency!

using UnPack
using KernelAbstractions 
using CUDA: @allowscalar
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!

const KA = KernelAbstractions

include("horizontal_advection.jl")

function computeLayerThicknessTendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        backend = KA.CPU())

    nthreads = 50
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendLayerThickness, Mesh.HorzMesh.PrimaryCells.nCells, ndrange=Mesh.HorzMesh.PrimaryCells.nCells)

    # compute horizontal advection of layer thickness
    horizontal_advection_tendency!(
        Tend, Prog, Diag, Mesh; backend = backend)
end

end
