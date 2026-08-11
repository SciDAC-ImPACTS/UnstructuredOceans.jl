import Adapt

"""
    VerticalMesh(mesh_fp, horzMesh; backend=KernelAbstractions.CPU())
    VerticalMesh(horzMesh; nVertLevels=1, backend=KernelAbstractions.CPU())

The vertical structure of the mesh: number of levels, the active-level index
fields, and the resting layer thicknesses (`restingThickness` and its column sum
`restingThicknessSum`, used to diagnose sea-surface height).

The first constructor reads the vertical grid from an MPAS mesh file `mesh_fp`
(paired with an already-read horizontal [`HorzMesh`](@ref)); the current
implementation supports stacked meshes where every column is full-depth. The
second constructor builds an `nVertLevels`-layer mesh from a horizontal mesh
alone and is intended for unit tests on periodic meshes, not real simulations.
`Adapt.adapt` moves the struct between host and device.
"""
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
    ActiveLevels{Edge}(mesh, maxLevelCell; backend=KA.CPU())

Reduce the per-cell active-level index `maxLevelCell` onto edges. `Top` is the
shallowest neighbouring cell (active on *both* sides), `Bot` the deepest (active
on *at least one* side); this mirrors MPAS's `maxLevelEdgeTop`/`maxLevelEdgeBot`.
Missing (index-0) neighbour cells on a bounded mesh are skipped so the reduction
never loads `maxLevelCell[0]`. The reduction is done on the host and the result
adapted to `backend`. On a stacked mesh (every cell full-depth) both come out
equal to `nVertLevels` everywhere, so the `for k in 1:maxLevelEdge.Top[iEdge]`
tendency loops cover the whole column.
"""
function ActiveLevels{Edge}(mesh, maxLevelCell; backend=KA.CPU())
    cellsOnEdge = Adapt.adapt(Array, mesh.Edges.cellsOnEdge)
    mlc         = Adapt.adapt(Array, maxLevelCell)
    nEdges      = mesh.Edges.nEdges

    Top = zeros(Int32, nEdges)
    Bot = zeros(Int32, nEdges)

    @inbounds for iEdge in 1:nEdges
        c1 = cellsOnEdge[1, iEdge]
        c2 = cellsOnEdge[2, iEdge]
        if c1 == 0
            Top[iEdge] = mlc[c2]
            Bot[iEdge] = mlc[c2]
        elseif c2 == 0
            Top[iEdge] = mlc[c1]
            Bot[iEdge] = mlc[c1]
        else
            Top[iEdge] = min(mlc[c1], mlc[c2])
            Bot[iEdge] = max(mlc[c1], mlc[c2])
        end
    end

    return ActiveLevels(Adapt.adapt(backend, Top), Adapt.adapt(backend, Bot))
end

"""
    ActiveLevels{Vertex}(mesh, maxLevelCell; backend=KA.CPU())

Reduce `maxLevelCell` onto dual-mesh vertices: `Top` is the shallowest and `Bot`
the deepest of the (up to `vertexDegree`) surrounding cells. Missing (index-0)
cells are skipped. See [`ActiveLevels{Edge}`](@ref).
"""
function ActiveLevels{Vertex}(mesh, maxLevelCell; backend=KA.CPU())
    cellsOnVertex = Adapt.adapt(Array, mesh.DualCells.cellsOnVertex)
    mlc           = Adapt.adapt(Array, maxLevelCell)
    nVertices     = mesh.DualCells.nVertices
    vertexDegree  = mesh.DualCells.vertexDegree

    Top = zeros(Int32, nVertices)
    Bot = zeros(Int32, nVertices)

    @inbounds for iVertex in 1:nVertices
        top = typemax(Int32)
        bot = zero(Int32)
        any_active = false
        for i in 1:vertexDegree
            iCell = cellsOnVertex[i, iVertex]
            iCell == 0 && continue
            any_active = true
            top = min(top, mlc[iCell])
            bot = max(bot, mlc[iCell])
        end
        if any_active
            Top[iVertex] = top
            Bot[iVertex] = bot
        end
    end

    return ActiveLevels(Adapt.adapt(backend, Top), Adapt.adapt(backend, Bot))
end

function VerticalMesh(mesh_fp, mesh; backend=KA.CPU())
    
    ds = NCDataset(mesh_fp, "r")
    
    # if uppercase(ds.attrib["is_periodic"]) != "YES"
    #     @warn "Non-periodic mesh detected; solid-wall boundary conditions assumed."
    # end
    
    nVertLevels = ds.dim["nVertLevels"]
    minLevelCell = ds["minLevelCell"][:]
    maxLevelCell = ds["maxLevelCell"][:]
    restingThickness = ds["restingThickness"][:,:,1]

    # check that the vertical mesh is stacked. Partial-cell / variable-column
    # masking (variable maxLevelCell) is deferred to a later phase; until then a
    # non-stacked mesh would be silently mis-simulated, so fail loudly instead.
    if !all(maxLevelCell .== nVertLevels)
        error("""(VerticalMesh initialization)
              Vertical mesh is not stacked: maxLevelCell is not uniformly \
              nVertLevels = $nVertLevels (found range \
              $(minimum(maxLevelCell))–$(maximum(maxLevelCell))). Variable-depth \
              columns need vertical masking, which is not implemented yet.""")
    end

    # Reduce the per-column active levels onto edges/vertices from the real
    # maxLevelCell so the tendency loops (for k in 1:maxLevelEdge.Top) cover the
    # whole column. On a stacked mesh these are nVertLevels everywhere.
    ActiveLevelsEdge   = ActiveLevels{Edge}(mesh, maxLevelCell; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh, maxLevelCell; backend=backend)

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
    # Unit thickness per layer, stacked full-depth: restingThickness is genuinely
    # (nVertLevels, nCells) and its column sum is nVertLevels per cell. For
    # nVertLevels == 1 this reduces to a column sum of 1.0, identical to before.
    restingThickness    = KA.ones(backend, Float64, nVertLevels, nCells)
    restingThicknessSum = sum(restingThickness; dims=1)

    # Reduce the (uniform) maxLevelCell onto edges/vertices; on this stacked test
    # mesh every active level equals nVertLevels.
    ActiveLevelsEdge   = ActiveLevels{Edge}(mesh, maxLevelCell; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh, maxLevelCell; backend=backend)

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

