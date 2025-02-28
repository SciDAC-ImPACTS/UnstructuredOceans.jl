
function horizontal_advection_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack dvEdge = Edges
    @unpack maxLevelEdge = VertMesh
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    normalVelocity = Prog.normalVelocity[end]
    # get the previous timesteps thicknessFlux (@Edges)
    @unpack layerThicknessEdge = Diag
    # unpack the layer thickness tendency term (@Cells)
    @unpack tendLayerThickness = Tend

    # initialize the kernel
    nthreads = 50
    kernel!  = thicknessFluxDivOnCell!(backend, nthreads)
    # use kernel to compute divergence of the thickness flux
    kernel!(tendLayerThickness,
            normalVelocity,
            layerThicknessEdge,
            nEdgesOnCell,
            edgesOnCell,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            areaCell,
            ndrange=nCells)

    # sync the backend
    KA.synchronize(backend)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendLayerThickness
end

@kernel function thicknessFluxDivOnCell!(tendency,
                                         @Const(normalVelocity),
                                         @Const(layerThicknessEdge),
                                         @Const(nEdgesOnCell),
                                         @Const(edgesOnCell),
                                         @Const(maxLevelEdgeTop),
                                         @Const(edgeSignOnCell),
                                         @Const(dvEdge),
                                         @Const(areaCell))

    iCell = @index(Global, Linear)

    # get inverse cell area
    invArea = 1. / areaCell[iCell]

    # create tmp varibale to store div reduction
    #div = @localmem eltype(DivCell) (1)

    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i,iCell]
        # loop over the number of (active) vertical layers
        for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k,iCell] += dvEdge[iEdge] * edgeSignOnCell[i,iCell] *
                                 normalVelocity[k, iEdge] *
                                 layerThicknessEdge[k, iEdge] *
                                 invArea
        end
    end
end
