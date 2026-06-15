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
    @synchronize()
end

@kernel function DivergenceOnCell_P2(DivCell,
                                     VecEdge,
                                     nEdgesOnCell,
                                     edgesOnCell,
                                     edgeSignOnCell,
                                     areaCell) #::Val{n}, where {n}

    iCell, k = @index(Global, NTuple)
    #iCell = @index(Global, Linear)
    #k = 1

    DivCell[k,iCell] = 0.0

    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        @inbounds DivCell[k,iCell] -= VecEdge[k,iEdge] * edgeSignOnCell[i,iCell]
    end

    DivCell[k,iCell] = DivCell[k,iCell] / areaCell[iCell]
    @synchronize()
end

function DivergenceOnCell!(DivCell, VecEdge, temp, Mesh::Mesh; backend=CUDABackend(), nthreads=50)
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack nVertLevels = VertMesh 
    @unpack dvEdge, nEdges = Edges
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells
    
    #nthreads = 50
    kernel1! = DivergenceOnCell_P1(backend, nthreads)
    kernel2! = DivergenceOnCell_P2(backend, nthreads)
    
    kernel1!(temp, VecEdge, dvEdge, nEdges, ndrange=(nEdges, nVertLevels))
    #kernel1!(temp, VecEdge, dvEdge, nEdges, ndrange=nEdges)
    
    kernel2!(DivCell,
             temp,
             nEdgesOnCell,
             edgesOnCell,
             edgeSignOnCell,
             areaCell,
             #ndrange=nCells)
             ndrange=(nCells, nVertLevels))

    KA.synchronize(backend)
    
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
                                dcEdge)
    # global indices over nEdges
    iEdge, k = @index(Global, NTuple)

    # TODO: add conditional statement to check for masking if needed

    # cell connectivity information for iEdge
    @inbounds @private jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds @private jCell2 = cellsOnEdge[2,iEdge]

    @inbounds GradEdge[k, iEdge] = (ScalarCell[k, jCell2] - ScalarCell[k, jCell1]) / dcEdge[iEdge]

    @synchronize()
end

function GradientOnEdge!(grad, hᵢ, Mesh::Mesh; backend=KA.CPU(), workgroupsize=64)
   
    @unpack HorzMesh, VertMesh = Mesh    

    @unpack Edges = HorzMesh
    @unpack nVertLevels = VertMesh 
    @unpack nEdges, dcEdge, cellsOnEdge = Edges
    
    kernel! = GradientOnEdge(backend)

    kernel!(grad, 
            hᵢ, 
            cellsOnEdge,
            dcEdge,
            workgroupsize=workgroupsize,
            ndrange=(nEdges, nVertLevels))

    KA.synchronize(backend)
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
    #iVertex = @index(Global, Linear)
    #k = 1
   
    #CurlVertex[k, iVertex] = 0.0

    @inbounds @private invAreaTriangle = 1.0 / areaTriangle[iVertex]

    for j in 1:vertexDegree
        @inbounds @private iEdge = edgesOnVertex[j, iVertex]

        @inbounds CurlVertex[k, iVertex] += dcEdge[iEdge] *
                                            invAreaTriangle *
                                            VecEdge[k, iEdge] *
                                            edgeSignOnVertex[j, iVertex]
    end

    @synchronize()
end

function CurlOnVertex!(CurlVertex, VecEdge, Mesh::Mesh; backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    

    @unpack nVertLevels, maxLevelVertex = VertMesh 
    @unpack DualCells, Edges = HorzMesh

    @unpack nEdges, dcEdge = Edges
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, edgeSignOnVertex, edgesOnVertex = DualCells

    nthreads = 50
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
           

    KA.synchronize(backend)
end

function interpolateCell2Edge!(edgeValue, cellValue, Mesh::Mesh;
                               backend = KA.CPU())
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack Edges = HorzMesh

    @unpack nVertLevels = VertMesh 
    @unpack nEdges, cellsOnEdge = Edges

    nthreads = 50
    kernel!  = interpolateCell2Edge(backend, nthreads)

    kernel!(edgeValue,
            cellValue,
            cellsOnEdge,
            nEdges,
            ndrange=nEdges)
            #ndrange=(nEdges, nVertLevels))

    KA.synchronize(backend)
end

@kernel function interpolateCell2Edge(edgeValue,
                                      cellValue,
                                      cellsOnEdge,
                                      arrayLength)
    # global indices over nEdges
    #iEdge, k = @index(Global, NTuple)
    iEdge = @index(Global, Linear)
    k = 1

    # TODO: add conditional statement to check for masking if needed

    # cell connectivity information for iEdge
    if iEdge < arrayLength + 1
        @inbounds @private iCell1 = cellsOnEdge[1,iEdge]      
        @inbounds @private iCell2 = cellsOnEdge[2,iEdge]

        @inbounds edgeValue[k, iEdge] = 0.5 * (cellValue[k, iCell1] +
                                            cellValue[k, iCell2])
    end

    @synchronize()
end

# Zeros out a vector along its entire length
@kernel function ZeroOutVector!(tendNormalVelocity, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        tendNormalVelocity[1, j] = 0.0
    end
    @synchronize()
end
