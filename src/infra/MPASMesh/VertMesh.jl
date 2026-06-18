import Adapt

mutable struct VerticalMesh{I, IV, FV, AL}
    nVertLevels::I

    minLevelCell::IV
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

"""
ActiveLevels constructor for a nVertLevel stacked *periodic* meshes
"""
function ActiveLevels(dim, eltype, backend, nVertLevels)
    Top = KA.ones(backend, eltype, dim) #.* eltype(nVertLevels)
    Bot = KA.ones(backend, eltype, dim) #.* eltype(nVertLevels)

    return ActiveLevels(Top, Bot)
end

function ActiveLevels{Edge}(mesh; backend=KA.CPU(), nVertLevels=1)
    return ActiveLevels(mesh.Edges.nEdges, Int32, backend, nVertLevels)
end

function ActiveLevels{Vertex}(mesh; backend=KA.CPU(), nVertLevels=1)
    ActiveLevels(mesh.DualCells.nVertices, Int32, backend, nVertLevels)
end

function VerticalMesh(mesh_fp, mesh; backend=KA.CPU())
    
    ds = NCDataset(mesh_fp, "r")
    
    if uppercase(ds.attrib["is_periodic"]) != "YES"
        @warn "Non-periodic mesh detected; solid-wall boundary conditions assumed."
    end
    
    nVertLevels = ds.dim["nVertLevels"]
    minLevelCell = ds["minLevelCell"][:]
    maxLevelCell = ds["maxLevelCell"][:]
    restingThickness = ds["restingThickness"][:,:,1]
    
    # check that the vertical mesh is stacked 
    if !all(maxLevelCell .== nVertLevels)
        @error """ (Vertical Mesh Initializaton)\n
               Vertical Mesh is not stacked. Must implement vertical masking
               before this mesh can be used
               """
    end

    
    ActiveLevelsEdge = ActiveLevels{Edge}(mesh; backend=backend,
                                          nVertLevels=nVertLevels)
    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh; backend=backend, 
                                              nVertLevels=nVertLevels)

    restingThicknessSum = sum(restingThickness; dims=1)

    VerticalMesh(nVertLevels,
                 Adapt.adapt(backend, minLevelCell),
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

    minLevelCell = KA.ones(backend, Int32, nCells)
    maxLevelCell = KA.ones(backend, Int32, nCells) .* Int32(nVertLevels)
    # unit thickness water column, irrespective of how many vertical levels
    restingThickness    = KA.ones(backend, Float64, nCells)
    restingThicknessSum = KA.ones(backend, Float64, nCells) # MIGHT NEED TO CHANGE THIS

    ActiveLevelsEdge = ActiveLevels{Edge}(mesh; backend=backend,
                                          nVertLevels=nVertLevels)

    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh; backend=backend, 
                                              nVertLevels=nVertLevels)

    # All array have been allocated on the requested backend,
    # so no need to call methods from Adapt
    VerticalMesh(nVertLevels,
                 minLevelCell,
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
                        Adapt.adapt(backend, x.minLevelCell), 
                        Adapt.adapt(backend, x.maxLevelCell),
                        Adapt.adapt(backend, x.maxLevelEdge),
                        Adapt.adapt(backend, x.maxLevelVertex),
                        Adapt.adapt(backend, x.restingThickness),
                        Adapt.adapt(backend, x.restingThicknessSum))
end

