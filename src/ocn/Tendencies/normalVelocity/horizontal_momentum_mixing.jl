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
                                              ::Type{Del2};
                                              viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
                                              nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, dvEdge = Edges
    @unpack cellsOnEdge, verticesOnEdge, boundaryEdge = Edges

    @unpack tendNormalVelocity = Tend
    @unpack velocityDivCell, relativeVorticity = Diag

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
            Val(VertMesh.nVertLevels),   # static vertical bound for a register-resident Enzyme tape
            ndrange=nEdges)

    # No host KA.synchronize: redundant on a single CUDA stream, and its
    # nonblocking sync worker segfaults Enzyme reverse mode (see UnstructuredOceansEnzymeExt).

    @pack! Tend = tendNormalVelocity
end

@kernel function horizontal_momentum_mixing_del2(tendency,
                                                  @Const(div),
                                                  @Const(relVort),
                                                  @Const(cellsOnEdge),
                                                  @Const(verticesOnEdge),
                                                  @Const(dcEdge),
                                                  @Const(dvEdge),
                                                  viscDel2,
                                                  @Const(boundaryEdge),
                                                  @Const(maxLevelEdgeTop),
                                                  ::Val{nVertLevels}) where {nVertLevels}

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds @private iCell1   = cellsOnEdge[1, iEdge]
        @inbounds @private iCell2   = cellsOnEdge[2, iEdge]
        @inbounds @private iVertex1 = verticesOnEdge[1, iEdge]
        @inbounds @private iVertex2 = verticesOnEdge[2, iEdge]

        @inbounds @private dcEdgeInv = 1.0 / dcEdge[iEdge]
        @inbounds @private dvEdgeInv = 1.0 / dvEdge[iEdge]

        # Static bound `nVertLevels` (a compile-time `Val`) instead of the runtime
        # `maxLevelEdgeTop[iEdge]`: keeps Enzyme's reverse tape in registers rather
        # than a per-thread device `malloc` (serialized global lock) that dominates
        # the GPU adjoint. The `k <= nLevels` guard reproduces the exact active-level
        # sum; a structured `if` (not an early `break`/`continue`) differentiates
        # correctly under Enzyme reverse.
        @inbounds nLevels = maxLevelEdgeTop[iEdge]
        for k in 1:nVertLevels
            if k <= nLevels
                @inbounds tendency[k, iEdge] += viscDel2[1] * (
                    (div[k, iCell2]    - div[k, iCell1])    * dcEdgeInv -
                    (relVort[k, iVertex2] - relVort[k, iVertex1]) * dvEdgeInv)
            end
        end
    end
end
