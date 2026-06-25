"""
methods for calculating tendencies of horizontal momentum diffusion using
KernelAbstractions
"""

abstract type MomentumDiffusion end

abstract type Del2 <: MomentumDiffusion end
abstract type Del4 <: MomentumDiffusion end

function horizontal_momentum_mixing_tendency!(Tend::TendencyVars,
                                              Prog::PrognosticVars,
                                              Diag::DiagnosticVars,
                                              Mesh::Mesh,
                                              ::Type{Del2})
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, dvEdge = Edges
    @unpack cellsOnEdge, verticesOnEdge, boundaryEdge = Edges
    viscDel2 = Edges.momentumDel2

    @unpack tendNormalVelocity = Tend
    @unpack velocityDivCell, relativeVorticity = Diag

    nthreads = 50
    kernel! = horizontal_momentum_mixing_del2(backend, nthreads)
    kernel!(tendNormalVelocity,
            velocityDivCell,
            relativeVorticity,
            cellsOnEdge,
            verticesOnEdge,
            dcEdge,
            dvEdge,
            viscDel2,
            boundaryEdge,
            maxLevelEdge.Top,
            ndrange=nEdges)

    # No host KA.synchronize: redundant on a single CUDA stream, and its
    # nonblocking sync worker segfaults Enzyme reverse mode (see MOKAEnzymeExt).

    @pack! Tend = tendNormalVelocity
end

@kernel function horizontal_momentum_mixing_del2(tendency,
                                                  @Const(div),
                                                  @Const(relVort),
                                                  @Const(cellsOnEdge),
                                                  @Const(verticesOnEdge),
                                                  @Const(dcEdge),
                                                  @Const(dvEdge),
                                                  @Const(viscDel2),
                                                  @Const(boundaryEdge),
                                                  @Const(maxLevelEdgeTop))

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds @private iCell1   = cellsOnEdge[1, iEdge]
        @inbounds @private iCell2   = cellsOnEdge[2, iEdge]
        @inbounds @private iVertex1 = verticesOnEdge[1, iEdge]
        @inbounds @private iVertex2 = verticesOnEdge[2, iEdge]

        @inbounds @private dcEdgeInv = 1.0 / dcEdge[iEdge]
        @inbounds @private dvEdgeInv = 1.0 / dvEdge[iEdge]

        for k in 1:maxLevelEdgeTop[iEdge]
            @inbounds tendency[k, iEdge] += viscDel2[1] * (
                (div[k, iCell2]    - div[k, iCell1])    * dcEdgeInv -
                (relVort[k, iVertex2] - relVort[k, iVertex1]) * dvEdgeInv)
        end
    end
end
