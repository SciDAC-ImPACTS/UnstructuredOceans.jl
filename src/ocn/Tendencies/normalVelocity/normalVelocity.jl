module NormalVelocity

export compute_normal_velocity_tendency!

using UnPack
using KernelAbstractions
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, ZeroOutVector!,
            DEFAULT_NTHREADS

# include tendency methods
include("pressure_gradient.jl")
include("horizontal_advection_and_coriolis.jl")
include("vector_invariant.jl")
include("horizontal_momentum_mixing.jl")
include("wind_forcing.jl")

"""
    compute_normal_velocity_tendency!(Tend, Prog, Diag, Mesh;
                                      coriolis=linearCoriolis, viscDel2=..., nthreads=...)

Assemble the normal-velocity tendency: pressure gradient + Coriolis/advection +
lateral mixing + wind forcing. `coriolis` selects the momentum-advection scheme
as a **type** (compile-time, so it is AD-safe as an Enzyme `Const`):
`linearCoriolis` (default; the ``\\sum w f u`` term used by the barotropic
gyre / IGW) or `vectorInvariant` (nonlinear ``q\\,\\mathbf{k}\\times(h\\mathbf{v})+\\nabla K``).
"""
function compute_normal_velocity_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        coriolis::Type{<:Coriolis}=linearCoriolis,
                                        viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
                                        nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(Tend.tendNormalVelocity)
    nEdges = Mesh.HorzMesh.Edges.nEdges
    nVertLevels = Mesh.VertMesh.nVertLevels
    kernel! = ZeroOutVector!(backend, nthreads)
    kernel!(Tend.tendNormalVelocity, nEdges, ndrange=(nEdges, nVertLevels))

    pressure_gradient_tendency!(
        Tend, Prog, Diag, Mesh, sshGradient; nthreads=nthreads)

    horizontal_advection_and_coriolis_tendency!(
        Tend, Prog, Diag, Mesh, coriolis; nthreads=nthreads)

    horizontal_momentum_mixing_tendency!(
        Tend, Prog, Diag, Mesh, Del2; viscDel2=viscDel2, nthreads=nthreads)

    wind_forcing_tendency!(
        Tend, Diag, Mesh; nthreads=nthreads)

end

end
