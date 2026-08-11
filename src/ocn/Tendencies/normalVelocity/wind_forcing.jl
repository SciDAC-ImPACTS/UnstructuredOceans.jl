function wind_forcing_tendency!(Tend::TendencyVars,
                                Diag::DiagnosticVars,
                                Mesh::Mesh;
                                nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack Edges = HorzMesh

    @unpack nEdges, windForcingEdge, boundaryEdge = Edges
    @unpack layerThicknessEdge = Diag
    @unpack tendNormalVelocity = Tend

    kernel! = wind_forcing_kernel!(backend, nthreads)
    kernel!(tendNormalVelocity,
            windForcingEdge,
            layerThicknessEdge,
            boundaryEdge,
            ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function wind_forcing_kernel!(tendency,
                                      windForcingEdge,
                                      layerThicknessEdge,
                                      boundaryEdge)

    iEdge = @index(Global, Linear)

    # Wind stress is a SURFACE flux: it is applied only to the top active layer
    # (level 1 for every stacked z-level column, where minLevelEdge == 1), unlike
    # the interior tendencies which loop over all k. Per-column surface indices
    # (minLevelEdge) only differ from 1 with partial cells, deferred to a later phase.
    if boundaryEdge[iEdge] != 1
        @inbounds tendency[1, iEdge] += windForcingEdge[iEdge] / layerThicknessEdge[1, iEdge]
    end
end
