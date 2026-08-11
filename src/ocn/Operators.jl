using KernelAbstractions

@doc raw"""
    DivergenceOnCell()

```math
\left[ \nabla \cdot \bm{F} \right]_i = \frac{1}{A}
                                       \sum_{e \in \rm{EC(i)}}
                                       n_{\rm e,i} F_{\rm e} l_{\rm e}
```
"""
@kernel function DivergenceOnCell_P1(temp, VecEdge, dvEdge, nEdges)

    iEdge, k = @index(Global, NTuple)
    if iEdge < nEdges + 1
        @inbounds temp[k,iEdge] = VecEdge[k,iEdge] * dvEdge[iEdge]
    end
    # @synchronize()
end

@kernel function DivergenceOnCell_P2(DivCell,
                                     VecEdge,
                                     nEdgesOnCell,
                                     edgesOnCell,
                                     edgeSignOnCell,
                                     areaCell) #::Val{n}, where {n}

    iCell, k = @index(Global, NTuple)

    DivCell[k,iCell] = 0.0

    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        @inbounds DivCell[k,iCell] -= VecEdge[k,iEdge] * edgeSignOnCell[i,iCell]
    end

    DivCell[k,iCell] = DivCell[k,iCell] / areaCell[iCell]
    # @synchronize()
end

@doc raw"""
    DivergenceOnCell!(DivCell, VecEdge, temp, Mesh; nthreads=DEFAULT_NTHREADS)

Compute the TRiSK discrete divergence of an edge-normal vector field `VecEdge`,
writing the per-cell result into `DivCell` (`temp` is edge-sized scratch):

```math
\left[ \nabla \cdot \bm{F} \right]_i = \frac{1}{A_i}
    \sum_{e \in \mathrm{EC}(i)} n_{e,i}\, F_e\, l_e
```

The sum is over the edges of cell ``i``, with ``A_i`` the cell area, ``l_e`` the
edge length, and ``n_{e,i}`` the edge sign. Launched as KernelAbstractions
kernels on the backend of `DivCell`.
"""
function DivergenceOnCell!(DivCell, VecEdge, temp, Mesh::Mesh; nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(DivCell)
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack nVertLevels = VertMesh 
    @unpack dvEdge, nEdges = Edges
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells
    
    kernel1! = DivergenceOnCell_P1(backend, nthreads)
    kernel2! = DivergenceOnCell_P2(backend, nthreads)
    
    kernel1!(temp, VecEdge, dvEdge, nEdges, ndrange=(nEdges, nVertLevels))
    
    kernel2!(DivCell,
             temp,
             nEdgesOnCell,
             edgesOnCell,
             edgeSignOnCell,
             areaCell,
             #ndrange=nCells)
             ndrange=(nCells, nVertLevels))
end

@doc raw"""
    GradientOnEdge()

```math
\left[ \nabla h \right]_e = \frac{1}{d_e} \sum_{i\in \rm{CE(e)}} -n_{\rm e,i} h_{\rm i}
```
    
"""
@kernel function GradientOnEdge(GradEdge,
                                ScalarCell,
                                cellsOnEdge,
                                dcEdge,
                                boundaryEdge)
    iEdge, k = @index(Global, NTuple)

    if boundaryEdge[iEdge] != 1
        @inbounds @private jCell1 = cellsOnEdge[1,iEdge]
        @inbounds @private jCell2 = cellsOnEdge[2,iEdge]

        @inbounds GradEdge[k, iEdge] = (ScalarCell[k, jCell2] - ScalarCell[k, jCell1]) / dcEdge[iEdge]
    end

    # @synchronize()
end

@doc raw"""
    GradientOnEdge!(grad, hᵢ, Mesh; workgroupsize=DEFAULT_NTHREADS)

Compute the TRiSK discrete gradient of a cell-centered scalar `hᵢ`, writing the
edge-normal result into `grad`:

```math
\left[ \nabla h \right]_e = \frac{h_{i_2(e)} - h_{i_1(e)}}{d_e}
```

where ``i_1(e), i_2(e)`` are the two cells adjacent to edge ``e`` and ``d_e`` the
distance between their centers. Boundary edges are skipped. Launched as a
KernelAbstractions kernel on the backend of `grad`.
"""
function GradientOnEdge!(grad, hᵢ, Mesh::Mesh; workgroupsize=DEFAULT_NTHREADS)
    backend = KA.get_backend(grad)
    @unpack HorzMesh, VertMesh = Mesh

    @unpack Edges = HorzMesh
    @unpack nVertLevels = VertMesh
    @unpack nEdges, dcEdge, cellsOnEdge, boundaryEdge = Edges

    kernel! = GradientOnEdge(backend)

    kernel!(grad,
            hᵢ,
            cellsOnEdge,
            dcEdge,
            boundaryEdge,
            workgroupsize=workgroupsize,
            ndrange=(nEdges, nVertLevels))
end

@kernel function CurlOnVertex(CurlVertex,
                              VecEdge,
                              edgesOnVertex,
                              dcEdge,
                              edgeSignOnVertex,
                              areaTriangle,
                              vertexDegree)

    # global indicies over nVertices and vertexDegree
    iVertex, k = @index(Global, NTuple)

    CurlVertex[k, iVertex] = 0.0

    @inbounds @private invAreaTriangle = 1.0 / areaTriangle[iVertex]

    for j in 1:vertexDegree
        @inbounds @private iEdge = edgesOnVertex[j, iVertex]

        # On a bounded mesh, boundary vertices have fewer than `vertexDegree` real
        # edges; the missing slots are stored as edgesOnVertex == 0 (see the mesh
        # setup loop in HorzMesh.jl, which skips them with `iEdge == 0 && continue`).
        # edgeSignOnVertex is 0 there, so the contribution is zero anyway — but the
        # loads dcEdge[0]/VecEdge[k,0] are still an out-of-bounds (index 0) read that
        # ROCm/HIP traps as an illegal address (CUDA silently read adjacent memory).
        # A structured `if iEdge != 0` (not `continue`) also differentiates correctly
        # under Enzyme; see the identical guard in the Coriolis tendency kernel.
        if iEdge != 0
            @inbounds CurlVertex[k, iVertex] += dcEdge[iEdge] *
                                                invAreaTriangle *
                                                VecEdge[k, iEdge] *
                                                edgeSignOnVertex[j, iVertex]
        end
    end

    # @synchronize()
end

@doc raw"""
    CurlOnVertex!(CurlVertex, VecEdge, Mesh; nthreads=DEFAULT_NTHREADS)

Compute the TRiSK discrete curl (relative vorticity) of an edge-normal vector
field `VecEdge` at dual-mesh vertices, writing the result into `CurlVertex`:

```math
\left[ \nabla \times \bm{v} \right]_v = \frac{1}{A_v}
    \sum_{e \in \mathrm{EV}(v)} t_{e,v}\, v_e\, d_e
```

where the sum is over the edges meeting at vertex ``v``, ``A_v`` is the dual
triangle area, ``d_e`` the cell-center distance, and ``t_{e,v}`` the edge sign.
Missing (boundary) edges are guarded. Launched as a KernelAbstractions kernel on
the backend of `CurlVertex`.
"""
function CurlOnVertex!(CurlVertex, VecEdge, Mesh::Mesh; nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(CurlVertex)
    @unpack HorzMesh, VertMesh = Mesh

    @unpack nVertLevels, maxLevelVertex = VertMesh
    @unpack DualCells, Edges = HorzMesh

    @unpack nEdges, dcEdge = Edges
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, edgeSignOnVertex, edgesOnVertex = DualCells

    kernel!  = CurlOnVertex(backend, nthreads)
    
    kernel!(CurlVertex,
            VecEdge,
            edgesOnVertex,
            dcEdge,
            edgeSignOnVertex, 
            areaTriangle,
            vertexDegree,
            #ndrange=nVertices)
            ndrange=(nVertices, nVertLevels))
end

function interpolateCell2Edge!(edgeValue, cellValue, Mesh::Mesh; nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(edgeValue)
    @unpack HorzMesh, VertMesh = Mesh
    @unpack Edges = HorzMesh

    @unpack nVertLevels = VertMesh
    @unpack nEdges, cellsOnEdge, boundaryEdge = Edges

    kernel!  = interpolateCell2Edge(backend, nthreads)

    kernel!(edgeValue,
            cellValue,
            cellsOnEdge,
            boundaryEdge,
            nEdges,
            ndrange=nEdges)
end

@kernel function interpolateCell2Edge(edgeValue,
                                      cellValue,
                                      cellsOnEdge,
                                      boundaryEdge,
                                      arrayLength)
    iEdge = @index(Global, Linear)
    k = 1

    if iEdge < arrayLength + 1
        if boundaryEdge[iEdge] != 1
            @inbounds @private iCell1 = cellsOnEdge[1,iEdge]
            @inbounds @private iCell2 = cellsOnEdge[2,iEdge]

            @inbounds edgeValue[k, iEdge] = 0.5 * (cellValue[k, iCell1] +
                                                    cellValue[k, iCell2])
        end
    end

    # @synchronize()
end

"""
    ZeroOutVector!(vector, arrayLength)

KernelAbstractions kernel that sets the first `arrayLength` entries of `vector`
(along its second dimension) to zero. Used to clear tendency accumulators before
summing contributions. Construct for a backend with
`ZeroOutVector!(backend, nthreads)` and launch with an `ndrange`.
"""
@kernel function ZeroOutVector!(tendNormalVelocity, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        tendNormalVelocity[1, j] = 0.0
    end
    # @synchronize()
end