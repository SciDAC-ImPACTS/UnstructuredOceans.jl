
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

    # initialize the kernel. Warp-multiple workgroup (DEFAULT_NTHREADS = 64) avoids
    # the idle lanes that a 50-wide group left in each 32-lane warp. Overridable via
    # the `nthreads` keyword.
    kernel!  = thicknessFluxDivOnCell!(backend, nthreads)
    # use kernel to compute divergence of the thickness flux
    kernel!(tendLayerThickness,
            thicknessFlux,
            nEdgesOnCell,     
            edgesOnCell,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            areaCell, 
            ndrange=nCells)

    # No host KA.synchronize: redundant on a single CUDA stream, and its
    # nonblocking sync worker segfaults Enzyme reverse mode (see MOKAEnzymeExt).

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendLayerThickness 
end

@kernel function thicknessFluxDivOnCell!(tendency,
                                         @Const(thicknessFlux),
                                         @Const(nEdgesOnCell),
                                         @Const(edgesOnCell),
                                         @Const(maxLevelEdgeTop),
                                         @Const(edgeSignOnCell),
                                         @Const(dvEdge),
                                         @Const(areaCell))

    iCell = @index(Global, Linear)

    # get inverse cell area
    @inbounds invArea = 1. / areaCell[iCell]

    # loop over number of edges in primary cell
    @inbounds for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        # dvEdge[iEdge], edgeSignOnCell[i,iCell] and invArea are all invariant in k:
        # fold them into one per-edge coefficient so the vertical loop is a single
        # load (thicknessFlux) + FMA per level instead of three loads and three
        # multiplies.
        @inbounds coef = dvEdge[iEdge] * edgeSignOnCell[i,iCell] * invArea
        # loop over the number of (active) vertical layers
        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k,iCell] += thicknessFlux[k,iEdge] * coef
        end
    end
end
