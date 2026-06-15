function wind_forcing_tendency!(Tend::TendencyVars,
                                Diag::DiagnosticVars,
                                Mesh::Mesh;
                                backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack Edges = HorzMesh

    @unpack nEdges, windForcingEdge, boundaryEdge = Edges
    @unpack layerThicknessEdge = Diag
    @unpack tendNormalVelocity = Tend

    nthreads = 50
    kernel! = wind_forcing_kernel!(backend, nthreads)
    kernel!(tendNormalVelocity,
            windForcingEdge,
            layerThicknessEdge,
            boundaryEdge,
            ndrange=nEdges)
    KA.synchronize(backend)

    @pack! Tend = tendNormalVelocity
end

@kernel function wind_forcing_kernel!(tendency,
                                      windForcingEdge,
                                      layerThicknessEdge,
                                      boundaryEdge)

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds tendency[1, iEdge] += windForcingEdge[iEdge] / layerThicknessEdge[1, iEdge]
    end
end
