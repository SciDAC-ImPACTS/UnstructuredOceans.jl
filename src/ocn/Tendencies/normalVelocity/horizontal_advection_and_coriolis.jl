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
                                                     ::Type{linearCoriolis})
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
    
    # initialize the kernel
    nthreads = 50
    kernel!  = coriolis_force_tendency_kernel!(backend, nthreads)
    # use kernel to compute coriolis and horizontal advection
    kernel!(tendNormalVelocity,
            normalVelocity,
            fᵉ, 
            nEdgesOnEdge, 
            edgesOnEdge,
            maxLevelEdge.Top, 
            weightsOnEdge, 
            ndrange = nEdges)
    # No host KA.synchronize: redundant on a single CUDA stream, and its
    # nonblocking sync worker segfaults Enzyme reverse mode (see MOKAEnzymeExt).

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end

@kernel function coriolis_force_tendency_kernel!(tendency,
                                                 normalVelocity,
                                                 fᵉ,
                                                 nEdgesOnEdge,
                                                 edgesOnEdge,
                                                 maxLevelEdgeTop,
                                                 weightsOnEdge)
    
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    @inbounds for i in 1:nEdgesOnEdge[iEdge]

        @inbounds eoe = edgesOnEdge[i,iEdge]

        # Use a structured `if` rather than `if eoe == 0 continue end`. Under
        # Enzyme reverse-mode the early-exit `continue` was not faithfully
        # replayed, so the adjoint executed the body with eoe == 0 and scattered
        # into normalVelocity[k, 0] — an out-of-bounds (index 0) write that
        # triggered ERROR_ILLEGAL_ADDRESS. A structured branch differentiates correctly.
        if eoe != 0
            @inbounds for k in 1:maxLevelEdgeTop[iEdge]
                tendency[k,iEdge] += weightsOnEdge[i,iEdge] *
                                     normalVelocity[k, eoe] *
                                     fᵉ[eoe]
            end
        end
    end
end
