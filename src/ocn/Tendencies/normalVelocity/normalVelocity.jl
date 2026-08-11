module NormalVelocity

export compute_normal_velocity_tendency!

using UnPack
using KernelAbstractions
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!,
            DEFAULT_NTHREADS

# include tendency methods
include("pressure_gradient.jl")
include("horizontal_advection_and_coriolis.jl")
include("horizontal_momentum_mixing.jl")
include("wind_forcing.jl")

function compute_normal_velocity_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
                                        nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(Tend.tendNormalVelocity)
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendNormalVelocity, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)

    pressure_gradient_tendency!(
        Tend, Prog, Diag, Mesh, sshGradient; nthreads=nthreads)

    horizontal_advection_and_coriolis_tendency!(
        Tend, Prog, Diag, Mesh, linearCoriolis; nthreads=nthreads)

    horizontal_momentum_mixing_tendency!(
        Tend, Prog, Diag, Mesh, Del2; viscDel2=viscDel2, nthreads=nthreads)

    wind_forcing_tendency!(
        Tend, Diag, Mesh; nthreads=nthreads)

end

end
