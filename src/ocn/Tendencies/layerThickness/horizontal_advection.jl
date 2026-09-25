
function horizontal_advection_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendLayerThickness)

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack dvEdge = Edges
    @unpack maxLevelEdge = VertMesh 
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    # get the previous timesteps thicknessFlux (@Edges)
    @unpack thicknessFlux = Diag
    # unpack the layer thickness tendency term (@Cells)
    @unpack tendLayerThickness = Tend 

    kernel!  = thicknessFluxDivOnCell!(backend, nthreads)
    kernel!(tendLayerThickness,
            thicknessFlux,
            nEdgesOnCell,     
            edgesOnCell,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            areaCell,
            Val(size(edgesOnCell, 1)),   # maxEdges as a compile-time constant
            Val(VertMesh.nVertLevels),   # static vertical bound
            ndrange=nCells)

    @pack! Tend = tendLayerThickness 
end

@kernel function thicknessFluxDivOnCell!(tendency,
                                         @Const(thicknessFlux),
                                         @Const(nEdgesOnCell),
                                         @Const(edgesOnCell),
                                         @Const(maxLevelEdgeTop),
                                         @Const(edgeSignOnCell),
                                         @Const(dvEdge),
                                         @Const(areaCell),
                                         ::Val{maxEdges},
                                         ::Val{nVertLevels}) where {maxEdges, nVertLevels}

    iCell = @index(Global, Linear)

    # get inverse cell area
    @inbounds invArea = 1. / areaCell[iCell]

    # Static bound `maxEdges` (a compile-time `Val`) instead of the runtime
    # `nEdgesOnCell[iCell]`: keeps Enzyme's reverse tape in registers rather than a
    # per-thread device `malloc` (serialized global lock) that dominates the GPU
    # adjoint. Padded slots have edgesOnCell == 0; a STRUCTURED `if` (not `continue`)
    # skips them — under Enzyme reverse a `continue` is mis-replayed and scatters into
    # index 0 (wrong gradient / OOB), per the Coriolis and CurlOnVertex guards.
    @inbounds for i in 1:maxEdges
        @inbounds iEdge = edgesOnCell[i,iCell]
        if iEdge != 0
            # dvEdge[iEdge], edgeSignOnCell[i,iCell] and invArea are all invariant in k:
            # fold them into one per-edge coefficient so the vertical loop is a single
            # load (thicknessFlux) + FMA per level instead of three loads and three
            # multiplies.
            @inbounds coef = dvEdge[iEdge] * edgeSignOnCell[i,iCell] * invArea
            # Static vertical bound (compile-time `Val`) + `k <= nLevels` guard so BOTH
            # loops are statically sized and Enzyme's tape stays in registers.
            @inbounds nLevels = maxLevelEdgeTop[iEdge]
            @inbounds for k in 1:nVertLevels
                if k <= nLevels
                    tendency[k,iCell] += thicknessFlux[k,iEdge] * coef
                end
            end
        end
    end
end
