import Adapt

using KernelAbstractions
const KA = KernelAbstractions

"""
    DiagnosticVars

Quantities recomputed from the [`PrognosticVars`](@ref) each timestep and reused
across the tendency calculations:

- `layerThicknessEdge` — layer thickness averaged from cell centers to edges
  ``[\\mathrm{m}]``, dimensioned `(nVertLevels, nEdges)`.
- `thicknessFlux` — thickness flux at edges, `(nVertLevels, nEdges)`.
- `velocityDivCell` — divergence of horizontal velocity ``[\\mathrm{s^{-1}}]``,
  `(nVertLevels, nCells)`.
- `relativeVorticity` — curl of horizontal velocity ``[\\mathrm{s^{-1}}]``,
  `(nVertLevels, nVertices)`.
- `kineticEnergyCell` — kinetic energy ``\\tfrac12|\\mathbf{u}|^2`` on cells
  ``[\\mathrm{m^2\\,s^{-2}}]``, `(nVertLevels, nCells)`.
- `layerThicknessVertex` — layer thickness interpolated to vertices
  ``[\\mathrm{m}]``, `(nVertLevels, nVertices)`.
- `potentialVorticityVertex` — potential vorticity ``(f+\\zeta)/h`` at vertices
  ``[\\mathrm{m^{-1}\\,s^{-1}}]``, `(nVertLevels, nVertices)`.

Construct with `DiagnosticVars(mesh; backend=...)`; `ocn_init` does this for you.
The core fields are filled by `diagnostic_compute!` inside [`ocn_timestep`](@ref);
the last three are filled only by the `vectorInvariant` momentum term (they stay
zero under the default `linearCoriolis`).
"""
mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}
    
    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges)
    layerThicknessEdge::FV2
    
    # var: ....
    # vim: (nVertLevels, nEdges)
    thicknessFlux::FV2

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    velocityDivCell::FV2

    # var: curl of horizontal velocity [s^{-1}]
    # dim: (nVertLevels, nVertices)
    relativeVorticity::FV2

    # --- Vector-invariant reconstructions (Phase 3) --------------------------
    # These are filled only by the `vectorInvariant` momentum term; with the
    # default `linearCoriolis` they stay zero and cost nothing, so the existing
    # cases are unaffected.

    # var: kinetic energy of horizontal velocity on cells [m^{2} s^{-2}]
    # dim: (nVertLevels, nCells)
    kineticEnergyCell::FV2

    # var: layer thickness interpolated from cells to vertices [m]
    # dim: (nVertLevels, nVertices)
    layerThicknessVertex::FV2

    # var: potential vorticity (f + relVort)/h at vertices [m^{-1} s^{-1}]
    # dim: (nVertLevels, nVertices)
    potentialVorticityVertex::FV2

    #= Performance Note:
    # ###########################################################
    #  While these can be stored as diagnostic variales I don't
    #  really think we need to do that. Only used locally within
    #  tendency calculations, so should be more preformant to
    #  calculate the values locally within the tendency loops.
    # ###########################################################

    # var: flux divergence [m s^{-1}] ?
    # dim: (nVertLevels, nCells)
    div_hu::Array{F,2}

    # var: Gradient of sea surface height at edges. [-]
    # dim: (nEdges), Time)?
    gradSSH::Array{F,1}
    =#

    function DiagnosticVars(layerThicknessEdge::AT2D,
                            thicknessFlux::AT2D,
                            velocityDivCell::AT2D,
                            relativeVorticity::AT2D,
                            kineticEnergyCell::AT2D,
                            layerThicknessVertex::AT2D,
                            potentialVorticityVertex::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, thicknessFlux,
                velocityDivCell, relativeVorticity,
                kineticEnergyCell, layerThicknessVertex,
                potentialVorticityVertex)

        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(layerThicknessEdge,
                        thicknessFlux,
                        velocityDivCell,
                        relativeVorticity,
                        kineticEnergyCell,
                        layerThicknessVertex,
                        potentialVorticityVertex)
    end
end

function DiagnosticVars(Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels

    # Here in the init function is where some sifting through will
    # need to be done, such that only diagnostic variables required by
    # the `Config` or requested by the `streams` will be activated.

    # create zero vectors to store diagnostic variables, on desired backend
    thicknessFlux = KA.zeros(backend, Float64, nVertLevels, nEdges)
    velocityDivCell = KA.zeros(backend, Float64, nVertLevels, nCells)
    relativeVorticity = KA.zeros(backend, Float64, nVertLevels, nVertices)
    layerThicknessEdge = KA.zeros(backend, Float64, nVertLevels, nEdges)
    kineticEnergyCell = KA.zeros(backend, Float64, nVertLevels, nCells)
    layerThicknessVertex = KA.zeros(backend, Float64, nVertLevels, nVertices)
    potentialVorticityVertex = KA.zeros(backend, Float64, nVertLevels, nVertices)

    DiagnosticVars(layerThicknessEdge,
                   thicknessFlux,
                   velocityDivCell,
                   relativeVorticity,
                   kineticEnergyCell,
                   layerThicknessVertex,
                   potentialVorticityVertex)
end

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.thicknessFlux),
                          Adapt.adapt(to, x.velocityDivCell),
                          Adapt.adapt(to, x.relativeVorticity),
                          Adapt.adapt(to, x.kineticEnergyCell),
                          Adapt.adapt(to, x.layerThicknessVertex),
                          Adapt.adapt(to, x.potentialVorticityVertex))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             nthreads=DEFAULT_NTHREADS)

    calculate_thickness_flux!(Diag, Prog, Mesh; nthreads=nthreads)
    calculate_velocity_div_cell!(Diag, Prog, Mesh; nthreads=nthreads)
    calculate_relative_vorticity!(Diag, Prog, Mesh; nthreads=nthreads)
    calculate_layer_thickness_edge!(Diag, Prog, Mesh; nthreads=nthreads)
end

#= Preformance Note:
   -----------------------------------------------------------------------
    Instead of `@unpack`ing and `@pack`ing the diagnostic field within the 
    `diagnostic_compute!` function would it be better to use a `@view`, 
    thereby reducing the array allocations? 
=# 

function calculate_layer_thickness_edge!(Diag::DiagnosticVars,
                                       Prog::PrognosticVars,
                                       Mesh::Mesh;
                                       nthreads=DEFAULT_NTHREADS)

    #layerThickness = Prog.layerThickness[:,:,end]
    @unpack layerThicknessEdge = Diag

    interpolateCell2Edge!(layerThicknessEdge,
                          Prog.layerThickness[end],
                          Mesh; nthreads=nthreads)

    @pack! Diag = layerThicknessEdge
end 

function calculate_thickness_flux!(Diag::DiagnosticVars,
                                  Prog::PrognosticVars,
                                  Mesh::Mesh;
                                  nthreads=DEFAULT_NTHREADS)

    backend = KA.get_backend(Diag.thicknessFlux)

    normalVelocity = Prog.normalVelocity[end]
    @unpack thicknessFlux, layerThicknessEdge = Diag

    nVertLevels = size(normalVelocity)[1]
    nEdges      = size(normalVelocity)[2]

    kernel!  = compute_thickness_flux!(backend, nthreads)

    kernel!(thicknessFlux, Prog.normalVelocity[end], layerThicknessEdge, nEdges, ndrange=(nEdges, nVertLevels))

    @pack! Diag = thicknessFlux
end

@kernel function compute_thickness_flux!(thicknessFlux,
                                        normalVelocity,
                                        layerThicknessEdge,
                                        arrayLength)

    # 2-D launch over (nEdges, nVertLevels): thickness flux at every level.
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds thicknessFlux[k,j] = normalVelocity[k,j] * layerThicknessEdge[k,j]
    end

    @synchronize()
end

function calculate_velocity_div_cell!(Diag::DiagnosticVars,
                                    Prog::PrognosticVars,
                                    Mesh::Mesh;
                                    nthreads=DEFAULT_NTHREADS)

    normalVelocity = Prog.normalVelocity[end]
    @unpack velocityDivCell, layerThicknessEdge = Diag

    DivergenceOnCell!(velocityDivCell, normalVelocity, layerThicknessEdge, Mesh; nthreads=nthreads)

    @pack! Diag = velocityDivCell
end

function calculate_relative_vorticity!(Diag::DiagnosticVars,
                                      Prog::PrognosticVars,
                                      Mesh::Mesh;
                                      nthreads=DEFAULT_NTHREADS)

    @unpack relativeVorticity = Diag

    CurlOnVertex!(relativeVorticity, Prog.normalVelocity[end], Mesh; nthreads=nthreads)

    @pack! Diag = relativeVorticity
end
