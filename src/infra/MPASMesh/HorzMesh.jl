using CUDA
using UnPack
using Accessors
using NCDatasets
using StructArrays

import Adapt
import KernelAbstractions as KA

###
### Types 
###

"""
    Cell

A type describing the location at the center of Primary Cell
"""
struct Cell end

"""
    Vertex

A type describing the location at the center of Dual Cells
"""
struct Vertex end

"""
    Edge

A type describing the location edge points where velocity is defined
"""
struct Edge end


### 
### Horizontal mesh struct
###

"""
    HorzMesh

A struct, comprised of SoA, describing a 2-D TRiSK mesh
"""
struct HorzMesh{PCT, DCT, ET}
    PrimaryCells::PCT
    DualCells::DCT
    Edges::ET
end

"""
"""
function Adapt.adapt_structure(backend, x::HorzMesh)
    return HorzMesh(Adapt.adapt(backend, x.PrimaryCells), 
                    Adapt.adapt(backend, x.DualCells),
                    Adapt.adapt(backend, x.Edges))
end

###
### Nodes of elements
###

# these are line segments
@kwdef struct Edges{I, FV, IV, FM, IM} 
    # I   --> (I)nt
    # FT  --> (F)loat (V)ector 
    # IT  --> (I)int  (V)ector 
    # FM  --> (F)loat (M)atrix  
    # IM  --> (I)int  (M)atrix  
    
    # dimension info
    nEdges::I

    # coordinate information
    xᵉ::FV   # X coordinates of edge midpoints in cartesian space
    yᵉ::FV   # Y coordinates of edge midpoints in cartesian space
    zᵉ::FV   # Z coordinates of edge midpoints in cartesian space
   
    # coriolis parameter
    fᵉ::FV 

    nEdgesOnEdge::IV # (n)umber of (E)dges (o) (E)dge

    # intra edge connectivity
    cellsOnEdge::IM    # (C)ells    (o)n (E)dge
    verticesOnEdge::IM # (V)ertices (o)n (E)dge

    # inter edge connectivity
    edgesOnEdge::IM   # (E)dges   (o)n (E)dge
    weightsOnEdge::FM # (W)eights (o)n (E)dge; reconstruction weights associated w/ EoE

    # area of edge
    angleEdge::FV
    
    # mask to determine if computation should be done on edge
    edgeMask::IM 

    # spacgin metrics
    dvEdge::FV
    dcEdge::FV 
end

###
### Elements of mesh 
###

# (P)rimary mesh cell
@kwdef struct PrimaryCells{I, FV, IV, IM} 
    # I   --> (I)nt
    # FV  --> (F)loat (V)ector 
    # IV  --> (I)int  (V)ector 
    # IM  --> (I)int  (M)atrix  
    
    # dimension info
    nCells::I
    maxEdges::I

    # coordinate information
    xᶜ::FV   # X coordinates of cell centers in cartesian space
    yᶜ::FV   # Y coordinates of cell centers in cartesian space
    zᶜ::FV   # Z coordinates of cell centers in cartesina space
    
    # coriolis parameter
    fᶜ::FV 

    nEdgesOnCell::IV # (n)umber of (E)dges (o)n (C)ell 

    # intra cell connectivity
    edgesOnCell::IM    # (E)dges    (o)n (C)ell; set of edges that define the boundary $P_i$
    verticesOnCell::IM # (V)ertices (o)n (C)ell

    # inter cell connectivity
    cellsOnCell::IM    # (C)ells    (o)n (C)ell
    edgeSignOnCell::IM # (E)dge (S)ign (o)n (C)ell; 

    # area of cell 
    areaCell::FV
end

# (D)ual mesh cell
@kwdef struct DualCells{I, FV, IM}
    # I   --> (I)nt
    # FV  --> (F)loat (V)ector 
    # IV  --> (I)int  (V)ector 
    # IM  --> (I)int  (M)atrix  
    
    # dimension info
    nVertices::I
    vertexDegree::I

    # coordinate information
    xᵛ::FV   # X coordinate of vertices in cartesian space
    yᵛ::FV   # Y coordinate of vertices in cartesian space
    zᵛ::FV   # Z coordinate of vertices in cartesina space
    
    # coriolis parameter
    fᵛ::FV 

    # intra vertex connecivity 
    edgesOnVertex::IM # (E)dges (o)n (V)ertex; set of edges that define the boundary $D_i$
    cellsOnVertex::IM # (C)ells (o)n (V)ertex 
    
    # inter vertex connecivity
    edgeSignOnVertex::IM #(E)dge (S)ign (o)n Vertex

    # area of triangle 
    areaTriangle::FV
end 

stack(arr, N) = [Tuple(arr[:,i]) for i in 1:N]

function readPrimaryMesh(ds)
    
    # get dimension info 
    nCells  = ds.dim["nCells"] 
    maxEdges = ds.dim["maxEdges"]

    # coordinate data 
    xᶜ = ds["xCell"][:]
    yᶜ = ds["yCell"][:]
    zᶜ = ds["zCell"][:]
    
    if haskey(ds, "fCell")
        fᶜ = ds["fCell"][:]
    else
        ## initalize coriolis as zero b/c not included in the base mesh
        fᶜ = zeros(eltype(xᶜ), nCells)
    end
    
    # intra connectivity
    edgesOnCell = ds["edgesOnCell"][:,:]
    verticesOnCell = ds["verticesOnCell"][:,:]

    # inter connectivity
    nEdgesOnCell = ds["nEdgesOnCell"][:]
    cellsOnCell = ds["cellsOnCell"][:,:]

    # edge sign on cell, empty for now will be popupated later
    edgeSignOnCell = zeros(eltype(nEdgesOnCell), (maxEdges, nCells))

    # Cell area
    areaCell = ds["areaCell"][:]

    PrimaryCells(nCells = nCells, maxEdges = maxEdges,
                 xᶜ = xᶜ, yᶜ = yᶜ, zᶜ = zᶜ, fᶜ = fᶜ, 
                 areaCell = areaCell,
                 edgesOnCell = edgesOnCell,
                 verticesOnCell = verticesOnCell,
                 cellsOnCell = cellsOnCell,
                 nEdgesOnCell = nEdgesOnCell,
                 edgeSignOnCell = edgeSignOnCell)
end

function readDualMesh(ds)

    # get dimension info 
    nVertices = ds.dim["nVertices"] 
    maxEdges  = ds.dim["maxEdges"]
    vertexDegree = ds.dim["vertexDegree"]

    # coordinate data 
    xᵛ = ds["xVertex"][:]
    yᵛ = ds["yVertex"][:]
    zᵛ = ds["zVertex"][:]
    
    if haskey(ds, "fVertex")
        fᵛ = ds["fVertex"][:]
    else
        # initalize coriolis as zero b/c not included in the base mesh
        fᵛ = zeros(eltype(xᵛ), nVertices)
    end

    # intra connectivity
    edgesOnVertex = ds["edgesOnVertex"][:,:]
    cellsOnVertex = ds["cellsOnVertex"][:,:]
    
    # get the integer type from the NetCDF file
    int_type = eltype(ds["edgesOnVertex"])
    # edge sign on vertex, empty for now will be popupated later
    edgeSignOnVertex = zeros(int_type, (maxEdges, nVertices))

    # Triangle area
    areaTriangle = ds["areaTriangle"][:]

    DualCells(nVertices = nVertices, vertexDegree = vertexDegree,
              xᵛ = xᵛ, yᵛ = yᵛ, zᵛ = zᵛ, fᵛ = fᵛ,
              edgesOnVertex = edgesOnVertex,
              cellsOnVertex = cellsOnVertex,
              edgeSignOnVertex = edgeSignOnVertex, 
              areaTriangle = areaTriangle)
end 

function readEdgeInfo(ds; nVertLevels = 1)

    # dimension data
    nEdges = ds.dim["nEdges"]
    
    # check that the requested `nVertLevels` is consistent with mesh file
    if haskey(ds.dim, "nVertLevels") && ds.dim["nVertLevels"] != nVertLevels
        @warn """Requested nVertLevels does not match the number of vertical \n
                 layers in the mesh file. 
              """ 
    end

    # base meshes don't neccessarily have vertical levels
    nVertLevels = haskey(ds.dim, "nVertLevels") ? ds.dim["nVertLevels"] : nVertLevels

    # coordinate data 
    xᵉ = ds["xEdge"][:]
    yᵉ = ds["yEdge"][:]
    zᵉ = ds["zEdge"][:]

    if haskey(ds, "fEdge")
        fᵉ = ds["fEdge"][:]
    else
        # initalize coriolis as zero b/c not included in the base mesh
        fᵉ = zeros(eltype(xᵉ), nEdges)
    end

    nEdgesOnEdge = ds["nEdgesOnEdge"][:]

    # intra connectivity
    cellsOnEdge = ds["cellsOnEdge"][:,:]
    verticesOnEdge = ds["verticesOnEdge"][:,:]

    # inter connectivity
    edgesOnEdge = ds["edgesOnEdge"][:,:]
    weightsOnEdge = ds["weightsOnEdge"][:,:]

    angleEdge = ds["angleEdge"][:]
    
    edgeMask = zeros(Int32, (nVertLevels, nEdges))

    dvEdge = ds["dvEdge"][:]
    dcEdge = ds["dcEdge"][:]

        
    Edges(nEdges = nEdges,
          xᵉ = xᵉ, yᵉ = yᵉ, zᵉ = zᵉ, fᵉ = fᵉ,
          nEdgesOnEdge = nEdgesOnEdge,
          cellsOnEdge = cellsOnEdge,
          verticesOnEdge = verticesOnEdge,
          edgesOnEdge = edgesOnEdge,
          weightsOnEdge = weightsOnEdge,
          angleEdge = angleEdge,
          edgeMask = edgeMask,
          dvEdge = dvEdge,
          dcEdge = dcEdge)
end

function signIndexField!(primaryCells::PrimaryCells, edges::Edges)
    
    @unpack cellsOnEdge = edges
    @unpack nCells, edgeSignOnCell, nEdgesOnCell, edgesOnCell = primaryCells

    @inbounds for iCell in 1:nCells, i in 1:nEdgesOnCell[iCell]
         
        iEdge = edgesOnCell[i, iCell]
        
        # vector points from cell 1 to cell 2
        if iCell == cellsOnEdge[1, iEdge]
            edgeSignOnCell[i, iCell] = -1
        else 
            edgeSignOnCell[i, iCell] = 1
        end 
    end    
    
    # PrimaryCell struct is immutable so need to use Accessor package,
    @reset primaryCells.edgeSignOnCell = edgeSignOnCell
end 

function signIndexField!(dualMesh::DualCells, edges::Edges)
    
    @unpack verticesOnEdge = edges
    @unpack vertexDegree, nVertices, edgeSignOnVertex, edgesOnVertex = dualMesh 

    for iVertex in 1:nVertices, i in 1:vertexDegree
         
        @inbounds iEdge = edgesOnVertex[i, iVertex]
        
        # if edge missing from vertex (i.e. vertex is on a culled boundary),
        # then leave edgeSignOnVertex as undefined (i.e. zero)
        if iEdge == 0 continue end

        # vector points from cell 1 to cell 2
        if iVertex == verticesOnEdge[1, iEdge]
            @inbounds edgeSignOnVertex[i, iVertex] = -1
        else 
            @inbounds edgeSignOnVertex[i, iVertex] = 1
        end 
    end    
    
    # DualCell struct is immutable so need to use Accessor package,
    @reset dualMesh.edgeSignOnVertex = edgeSignOnVertex
end 

function setBoundaryMask!(edges::Edges, v_mesh)
    
    @unpack Bot, Top = v_mesh.maxLevelEdge
    @unpack nEdges, edgeMask = edges
    
    for iEdge in 1:nEdges, k in Bot[iEdge]:Top[iEdge]
        edgeMask[k, iEdge] = 1.0
    end

    # Edges struct is immutable so need to use Accessor package,
    @reset edges.edgeMask = edgeMask
end

function ReadHorzMesh(meshPath::String; nVertLevels=1, backend=KA.CPU())
    
    ds = NCDataset(meshPath, "r", format=:netcdf4)

    PrimaryMesh = readPrimaryMesh(ds)
    DualMesh    = readDualMesh(ds)
    edges       = readEdgeInfo(ds; nVertLevels=nVertLevels)
    
    # set the edge sign on cells (primary mesh)
    signIndexField!(PrimaryMesh, edges)
    # set the edge sign on vertices (dual mesh)
    signIndexField!(DualMesh, edges)
    
    mesh = HorzMesh(PrimaryMesh, DualMesh, edges)
    
    # move the horizontal mesh struct to requested backend
    # NOTE: this does not happen earlier (i.e. in constructors of PrimaryCells,
    #       DualCells, Edges) b/c of the uninitialized fields in the mesh 
    #       strucutres which are populated above. It's more efficent to have
    #       those calculations happen on CPU (I think)
    Adapt.adapt_structure(backend, mesh)
end

function Adapt.adapt_structure(to, edges::Edges)
    return Edges(nEdges = edges.nEdges,
                 xᵉ = Adapt.adapt(to, edges.xᵉ),
                 yᵉ = Adapt.adapt(to, edges.yᵉ),
                 zᵉ = Adapt.adapt(to, edges.zᵉ),
                 fᵉ = Adapt.adapt(to, edges.fᵉ),
                 nEdgesOnEdge = Adapt.adapt(to, edges.nEdgesOnEdge),
                 cellsOnEdge = Adapt.adapt(to, edges.cellsOnEdge),
                 verticesOnEdge = Adapt.adapt(to, edges.verticesOnEdge),
                 edgesOnEdge = Adapt.adapt(to, edges.edgesOnEdge),
                 weightsOnEdge = Adapt.adapt(to, edges.weightsOnEdge),
                 angleEdge = Adapt.adapt(to, edges.angleEdge),
                 edgeMask = Adapt.adapt(to, edges.edgeMask),
                 dvEdge = Adapt.adapt(to, edges.dvEdge),
                 dcEdge = Adapt.adapt(to, edges.dcEdge))
end

function Adapt.adapt_structure(to, cells::PrimaryCells)
    return PrimaryCells(nCells = cells.nCells,
                        maxEdges = cells.maxEdges,
                        xᶜ = Adapt.adapt(to, cells.xᶜ),
                        yᶜ = Adapt.adapt(to, cells.yᶜ),
                        zᶜ = Adapt.adapt(to, cells.zᶜ),
                        fᶜ = Adapt.adapt(to, cells.fᶜ),
                        nEdgesOnCell = Adapt.adapt(to, cells.nEdgesOnCell),
                        edgesOnCell = Adapt.adapt(to, cells.edgesOnCell),
                        verticesOnCell = Adapt.adapt(to, cells.verticesOnCell),
                        cellsOnCell = Adapt.adapt(to, cells.cellsOnCell),
                        edgeSignOnCell = Adapt.adapt(to, cells.edgeSignOnCell),
                        areaCell = Adapt.adapt(to, cells.areaCell))
end

function Adapt.adapt_structure(to, duals::DualCells)
    return DualCells(nVertices = duals.nVertices,
                     vertexDegree = duals.vertexDegree,
                     xᵛ = Adapt.adapt(to, duals.xᵛ),
                     yᵛ = Adapt.adapt(to, duals.yᵛ),
                     zᵛ = Adapt.adapt(to, duals.zᵛ),
                     fᵛ = Adapt.adapt(to, duals.fᵛ),
                     edgesOnVertex = Adapt.adapt(to, duals.edgesOnVertex),
                     cellsOnVertex = Adapt.adapt(to, duals.cellsOnVertex),
                     edgeSignOnVertex = Adapt.adapt(to, duals.edgeSignOnVertex),
                     areaTriangle = Adapt.adapt(to, duals.areaTriangle))
end
