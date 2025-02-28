module normalVelocity

export computeNormalVelocityTendency!

using UnPack
using KernelAbstractions
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, ForcingVars, Mesh, GlobalConfig, ZeroOutVector!

#const KA = KernelAbstractions

# include tendecy methods
include("velocity_forcing.jl")
include("pressure_gradient.jl")
include("horizontal_momentum_mixing.jl")
include("horizontal_advection_and_coriolis.jl")

### TO DO:
#
#  [ ] Parser functions that will iterate over config files to dispatch
#      the proper tendency type.
####

function computeNormalVelocityTendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh,
                                        Forcing::ForcingVars,
                                        Config::GlobalConfig;
                                        backend = CUDABackend())

    nthreads = 50
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendNormalVelocity, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)
    #=
    # hard code the pressure gradient as SSH Gradient for now, in the future
    # we will want some functionality to parse config (e.g.):
    #
    #       pGradType = parsing_function(Config)
    =#
    pGradType = sshGradient

    # compute pressure gradient tendency on requested backend
    pressure_gradient_tendency!(
        Tend, Prog, Diag, Mesh, pGradType; backend = backend
       )

    # hard coded type for now, see above about inquiring into the config struct
    coriolisType = linearCoriolis

    # compute horizontal advection and corilois tendency on requested backend
    horizontal_advection_and_coriolis_tendency!(
        Tend, Prog, Diag, Mesh, coriolisType; backend = backend
       )

    # compute horizontal momentum mixing
    horizontal_momentum_mixing_tendency!(
        Tend, Prog, Diag, Mesh, Del2; backend = backend
    )

    # add forcing
    velocity_forcing_tendency!(
        Tend, Prog, Diag, Mesh, Forcing; backend = backend
    )
end

end
