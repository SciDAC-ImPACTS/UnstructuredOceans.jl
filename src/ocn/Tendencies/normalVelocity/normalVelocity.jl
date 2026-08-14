module normalVelocity

export computeNormalVelocityTendency!

using UnPack
using KernelAbstractions
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!

# include tendency methods
include("pressure_gradient.jl")
include("horizontal_advection_and_coriolis.jl")
include("horizontal_momentum_mixing.jl")
include("wind_forcing.jl")

function computeNormalVelocityTendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh)
    backend = KernelAbstractions.get_backend(Tend.tendNormalVelocity)
    nthreads = 50
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendNormalVelocity, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)

    pressure_gradient_tendency!(
        Tend, Prog, Diag, Mesh, sshGradient)

    horizontal_advection_and_coriolis_tendency!(
        Tend, Prog, Diag, Mesh, linearCoriolis)

    horizontal_momentum_mixing_tendency!(
        Tend, Prog, Diag, Mesh, Del2)

    wind_forcing_tendency!(
        Tend, Diag, Mesh)

end

end
