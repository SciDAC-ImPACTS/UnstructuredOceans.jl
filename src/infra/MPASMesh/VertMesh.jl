using OffsetArrays
import Adapt

mutable struct VerticalMesh{I, IV, FV, AL}
    nVertLevels::I

    maxLevelCell::IV

    maxLevelEdge::AL
    maxLevelVertex::AL

    # var: Layer thickness when the ocean is at rest [m]
    # dim: (nVertLevels, nCells)
    restingThickness::FV
    # var: Total thickness when the ocean is at rest [m]
    # dim: (1, nCells)
    restingThicknessSum::FV
end

mutable struct ActiveLevels{IV}
    # Index to the last {edge|vertex} in a column with active ocean cells 
    # on *all* sides of it
    Top::IV 
    # Index to the last {edge|vertex} in a column with at least one active
    # ocean cell around it
    Bot::IV
end

function padded_index_array(dimLength; backend=KA.CPU, eltype=Int32)
    OffsetArray(KA.zeros(backend, eltype, dimLength + 1), 0:dimLength)
end

"""
ActiveLevels constructor for a nVertLevel stacked *periodic* meshes
"""
function ActiveLevels(dim, eltype, backend)
    Top = KA.ones(backend, eltype, dim)
    Bot = KA.ones(backend, eltype, dim)

    return ActiveLevels(Top, Bot)
end

function ActiveLevels{Edge}(maxLevelCell, h_mesh; backend=KA.CPU())
    
    @unpack nEdges, cellsOnEdge = h_mesh.Edges

    # Top is the minimum (shallowest) of the surrounding cells
    Top = padded_index_array(nEdges; backend=backend) 
    # Bot is the maximum (deepest) of the surrounding cells
    Bot = padded_index_array(nEdges; backend=backend)
    
    for iEdge in 1:nEdges
        @inbounds iCell1 = cellsOnEdge[1, iEdge] 
        @inbounds iCell2 = cellsOnEdge[2, iEdge] 

        Top[iEdge] = min(maxLevelCell[iCell1], maxLevelCell[iCell2])
        Bot[iEdge] = max(maxLevelCell[iCell1], maxLevelCell[iCell2])
    end

    return ActiveLevels(Top, Bot)
end

function ActiveLevels{Vertex}(maxLevelCell, h_mesh; backend=KA.CPU())

    @unpack nVertices, cellsOnVertex, vertexDegree = h_mesh.DualCells

    # Top is the minimum (shallowest) of the surrounding cells
    Top = padded_index_array(nVertices; backend=backend) 
    # Bot is the maximum (deepest) of the surrounding cells
    Bot = padded_index_array(nVertices; backend=backend)

    for iVertex in 1:nVertices
        # get vector indices of the cellsOnVertex (e.g. (3,))
        cellsOnVertex_i = [cellsOnVertex[i, iVertex] for i in 1:vertexDegree]

        Top[iVertex] = minimum(maxLevelCell[cellsOnVertex_i])
        Bot[iVertex] = maximum(maxLevelCell[cellsOnVertex_i])
    end

    return ActiveLevels(Top, Bot)
end

function VerticalMesh(mesh_fp, mesh; backend=KA.CPU())
    
    ds = NCDataset(mesh_fp, "r")
    
    nCells = mesh.PrimaryCells.nCells
    # Pre-allocate zero indexed offsetarrays 
    maxLevelCell = padded_index_array(nCells; backend=backend)

    nVertLevels = ds.dim["nVertLevels"]
    # ....
    maxLevelCell[1:end] = ds["maxLevelCell"][:]
    restingThickness = ds["restingThickness"][:,:,1]
    
    # check that the vertical mesh is stacked 
    if !all(maxLevelCell[1:end] .== nVertLevels)
        @error """ (Vertical Mesh Initializaton)\n
               Vertical Mesh is not stacked. Must implement vertical masking
               before this mesh can be used
               """
    end

    
    ActiveLevelsEdge = ActiveLevels{Edge}(maxLevelCell, mesh; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(maxLevelCell, mesh; backend=backend)

    restingThicknessSum = sum(restingThickness; dims=1)

    VerticalMesh(nVertLevels,
                 Adapt.adapt(backend, maxLevelCell),
                 ActiveLevelsEdge,
                 ActiveLevelsVertex, 
                 Adapt.adapt(backend, restingThickness),
                 Adapt.adapt(backend, restingThicknessSum))
end

"""
Constructor for an (n) layer stacked vertical mesh. Only valid when paired 
with a *periodic* horizontal mesh.

This function is handy for unit test that read in purely horizontal meshes. 

NOTE: Not to be used for real simualtions, only for unit testing. 
"""
function VerticalMesh(mesh; nVertLevels=1, backend=KA.CPU())

    nCells = mesh.PrimaryCells.nCells

    maxLevelCell = KA.ones(backend, Int32, nCells) .* Int32(nVertLevels)
    # unit thickness water column, irrespective of how many vertical levels
    restingThickness    = KA.ones(backend, Float64, nCells)
    restingThicknessSum = KA.ones(backend, Float64, nCells) # MIGHT NEED TO CHANGE THIS

    ActiveLevelsEdge = ActiveLevels{Edge}(maxLevelCell, mesh; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(maxLevelCell, mesh; backend=backend)

    # All array have been allocated on the requested backend,
    # so no need to call methods from Adapt
    VerticalMesh(nVertLevels,
                 maxLevelCell,
                 ActiveLevelsEdge,
                 ActiveLevelsVertex, 
                 restingThickness,
                 restingThicknessSum)
end

function Adapt.adapt_structure(backend, x::ActiveLevels)
    return ActiveLevels(Adapt.adapt(backend, x.Top), 
                        Adapt.adapt(backend, x.Bot))
end

function Adapt.adapt_structure(backend, x::VerticalMesh)
    return VerticalMesh(x.nVertLevels,
                        Adapt.adapt(backend, x.maxLevelCell),
                        Adapt.adapt(backend, x.maxLevelEdge),
                        Adapt.adapt(backend, x.maxLevelVertex),
                        Adapt.adapt(backend, x.restingThickness),
                        Adapt.adapt(backend, x.restingThicknessSum))
end

