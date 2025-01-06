"""
methods for calculting tendencies of horizontal momentum diffusion using 
KernelAbstractions 
"""

abstract type MomentumDiffusion end 

abstract type Del2 <: MomentumDiffusion end 
abstract type Del4 <: MomentumDiffusion end 

function horizontal_momentum_mixing_tendency!(Tend::TendencyVars,
                                              Prog::PrognosticVars,
                                              Diag::DiagnosticVars,
                                              Mesh::Mesh, 
                                              ::Type{Del2}; 
                                              backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, dcEdge, dvEdge = Edges
    @unpack cellsOnEdge, verticesOnEdge, edgeMask = Edges

    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    # get needed fields from diagnostics structure
    @unpack velocityDivCell, relativeVorticity = Diag

    viscDel2 = 1.0

    # initialize the kernel
    kernel! = SSHGradOnEdge!(backend)
    # use kernel to compute horizontal momentum mixing
    kernel!(tendNormalVelocity,
            velocityDivCell, 
            relativeVorticity, 
            cellsOnEdge, 
            verticesOnEdge,
            dcEdge,
            dvEdge,
            viscDel2, # WHERE IS THIS COMING FROM?
            maxLevelEdge.Top,
            edgeMask,
            ndrange=nEdges)

    # sync the backend 
    KA.synchronize(backend)
    
    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity 
end

@kernel function horizontalm_momentum_mixing_del2(tendency, 
                                                  @Const(div),
                                                  @Const(relVort),
                                                  @Const(cellsOnEdge),
                                                  @Const(verticesOnEdge), 
                                                  @Const(dcEdge), 
                                                  @Const(dvEdge), 
                                                  @Const(viscDel2),
                                                  @Const(maxLevelEdgeTop), 
                                                  @Const(edgeMask))
    # global indices over nEdges
    iEdge = @index(Global, Linear)
    
    @inbounds @private iCell1 = cellsOnEdge[1, iEdge]
    @inbounds @private iCell2 = cellsOnEdge[2, iEdge]
    @inbounds @private iVertex1 = verticesOnEdge[1, iEdge]
    @inbounds @private iVertex2 = verticesOnEdge[2, iEdge]

    @inbounds @private dcEdgeInv = 1.0 / dcEdgeInv[iEdge] 
    @inbounds @private dvEdgeInv = 1.0 / dvEdgeInv[iEdge] 

    for k in 1:maxLevelEdgeTop[iEdge]
        @inbounds tendency[k, iEdge] += (
            (div[k, iCell2] - div[k, iCell1]) * dcEdgeInv -
            (relVort[k, iVertex2] - relVort[k, iVertex1]) * dvEdgeInv) *
            viscDel2 * edgeMask[k, iEdge]
    end
end
