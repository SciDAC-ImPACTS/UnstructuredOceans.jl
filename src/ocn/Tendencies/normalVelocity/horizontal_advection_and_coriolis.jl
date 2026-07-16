"""
methods for calculating tendencies from the horizontal advection and
coriolis force using KernelAbstractions
"""

# might need some help determining the way to abstract the vorticity options
abstract type Coriolis end

# define the supported coriolis formulations to dispatch on
abstract type linearCoriolis <: Coriolis end

function horizontal_advection_and_coriolis_tendency!(Tend::TendencyVars,
                                                     Prog::PrognosticVars,
                                                     Diag::DiagnosticVars,
                                                     Mesh::Mesh,
                                                     ::Type{linearCoriolis};
                                                     nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, nEdgesOnEdge = Edges
    @unpack weightsOnEdge, fᵉ, cellsOnEdge, edgesOnEdge = Edges

    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity[end]
    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    
    kernel!  = coriolis_force_tendency_kernel!(backend, nthreads)
    kernel!(tendNormalVelocity,
            normalVelocity,
            fᵉ, 
            nEdgesOnEdge, 
            edgesOnEdge,
            maxLevelEdge.Top, 
            weightsOnEdge, 
            ndrange = nEdges)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end

@kernel function coriolis_force_tendency_kernel!(tendency,
                                                 @Const(normalVelocity),
                                                 @Const(fᵉ),
                                                 @Const(nEdgesOnEdge),
                                                 @Const(edgesOnEdge),
                                                 @Const(maxLevelEdgeTop),
                                                 @Const(weightsOnEdge))
    
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    # maxLevelEdgeTop[iEdge] is invariant across the neighbour loop — load it once.
    @inbounds nLevels = maxLevelEdgeTop[iEdge]

    # edgesOnEdge / weightsOnEdge are stored EDGE-MAJOR ([iEdge, i]) so that, with the
    # thread index iEdge as the leading (unit-stride) dimension, a warp's reads of a
    # fixed neighbour i are contiguous and coalesce into few cache lines (see the layout
    # note in readEdgeInfo). Indexing them [i, iEdge] here would reintroduce the strided,
    # L2-spilling access that made this kernel scale super-linearly on the GPU.
    @inbounds for i in 1:nEdgesOnEdge[iEdge]

        @inbounds eoe = edgesOnEdge[iEdge,i]

        # Use a structured `if` rather than `if eoe == 0 continue end`. Under
        # Enzyme reverse-mode the early-exit `continue` was not faithfully
        # replayed, so the adjoint executed the body with eoe == 0 and scattered
        # into normalVelocity[k, 0] — an out-of-bounds (index 0) write that
        # triggered ERROR_ILLEGAL_ADDRESS. A structured branch differentiates correctly.
        if eoe != 0
            # weightsOnEdge[iEdge,i] and fᵉ[eoe] are both invariant in k: fold them
            # into one per-neighbour coefficient outside the vertical loop so each
            # level does a single load (normalVelocity) + FMA instead of two loads
            # and two multiplies.
            @inbounds coef = weightsOnEdge[iEdge,i] * fᵉ[eoe]
            @inbounds for k in 1:nLevels
                tendency[k,iEdge] += coef * normalVelocity[k, eoe]
            end
        end
    end
end
