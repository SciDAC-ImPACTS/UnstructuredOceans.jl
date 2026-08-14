"""
Surface / external forcing of the normal-velocity tendency, dispatched by a
forcing type so new forcings (restoring, tidal, surface fluxes) can be added as a
method + a config flag without editing the assembly chain. `WindForcing` is the
bulk wind stress; its data lives in [`ForcingVars`](@ref) on the `Mesh`.
"""
abstract type AbstractForcing end

# Bulk wind-stress forcing (surface momentum flux).
abstract type WindForcing <: AbstractForcing end

function forcing_tendency!(Tend::TendencyVars,
                           Diag::DiagnosticVars,
                           Mesh::Mesh,
                           ::Type{WindForcing};
                           nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh = Mesh
    @unpack Edges = HorzMesh

    @unpack nEdges, boundaryEdge = Edges
    windStressEdge = Mesh.Forcing.windStressEdge
    @unpack layerThicknessEdge = Diag
    @unpack tendNormalVelocity = Tend

    kernel! = wind_forcing_kernel!(backend, nthreads)
    kernel!(tendNormalVelocity,
            windStressEdge,
            layerThicknessEdge,
            boundaryEdge,
            ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function wind_forcing_kernel!(tendency,
                                      windStressEdge,
                                      layerThicknessEdge,
                                      boundaryEdge)

    iEdge = @index(Global, Linear)

    # Wind stress is a SURFACE flux: it is applied only to the top active layer
    # (level 1 for every stacked z-level column, where minLevelEdge == 1), unlike
    # the interior tendencies which loop over all k. Per-column surface indices
    # (minLevelEdge) only differ from 1 with partial cells, deferred to a later phase.
    if boundaryEdge[iEdge] != 1
        @inbounds tendency[1, iEdge] += windStressEdge[iEdge] / layerThicknessEdge[1, iEdge]
    end
end
