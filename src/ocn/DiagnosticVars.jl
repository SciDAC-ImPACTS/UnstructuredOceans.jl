import Adapt

using CUDA
using KernelAbstractions

mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}
    
    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges)
    layerThicknessEdge::FV2
    
    # var: ....
    # vim: (nVertLevels, nEdges)
    thicknessFlux::FV2

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    velocityDivCell::FV2

    # var: curl of horizontal velocity [s^{-1}]
    # dim: (nVertLevels, nVertices)
    relativeVorticity::FV2

    #= Performance Note: 
    # ###########################################################
    #  While these can be stored as diagnostic variales I don't 
    #  really think we need to do that. Only used locally within 
    #  tendency calculations, so should be more preformant to 
    #  calculate the values locally within the tendency loops. 
    # ###########################################################
     
    # var: flux divergence [m s^{-1}] ? 
    # dim: (nVertLevels, nCells)
    div_hu::Array{F,2}
    
    # var: Gradient of sea surface height at edges. [-] 
    # dim: (nEdges), Time)?
    gradSSH::Array{F,1}
    =#
    
    #= UNUSED FOR NOW:
    # var: horizontal velocity, tangential to an edge [m s^{-1}] 
    # dim: (nVertLevels, nEdges)
    tangentialVelocity::Array{F, 2}

    # var: kinetic energy of horizonal velocity on cells [m^{2} s^{-2}]
    # dim: (nVertLevels, nCells)
    kineticEnergyCell::Array{F, 2}

    =# 

    function DiagnosticVars(layerThicknessEdge::AT2D, 
                            thicknessFlux::AT2D, 
                            velocityDivCell::AT2D, 
                            relativeVorticity::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, thicknessFlux,
                velocityDivCell, relativeVorticity)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(layerThicknessEdge,
                        thicknessFlux,
                        velocityDivCell,
                        relativeVorticity)
    end
end 
 
function DiagnosticVars(config::GlobalConfig, Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels
    
    # Here in the init function is where some sifting through will 
    # need to be done, such that only diagnostic variables required by 
    # the `Config` or requested by the `streams` will be activated. 
    
    FT = Float64

    # create zero vectors to store diagnostic variables, on desired backend
    thicknessFlux = KA.zeros(backend, FT, nVertLevels, nEdges) 
    velocityDivCell = KA.zeros(backend, FT, nVertLevels, nCells)
    relativeVorticity = KA.zeros(backend, FT, nVertLevels, nVertices)
    # initialize to Inf to avoid divide by zero and NaN problems 
    layerThicknessEdge = KA.ones(backend, FT, nVertLevels, nEdges) * -typemax(FT)

    DiagnosticVars(layerThicknessEdge,
                   thicknessFlux,
                   velocityDivCell,
                   relativeVorticity)
end 

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.thicknessFlux), 
                          Adapt.adapt(to, x.velocityDivCell),
                          Adapt.adapt(to, x.relativeVorticity))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             backend = KA.CPU())

    calculate_thicknessFlux!(Diag, Prog, Mesh; backend = backend)
    calculate_velocityDivCell!(Diag, Prog, Mesh; backend = backend)
    calculate_relativeVorticity!(Diag, Prog, Mesh; backend = backend)
    calculate_layerThicknessEdge!(Diag, Prog, Mesh; backend = backend)
end 

#= Preformance Note:
   -----------------------------------------------------------------------
    Instead of `@unpack`ing and `@pack`ing the diagnostic field within the 
    `diagnostic_compute!` function would it be better to use a `@view`, 
    thereby reducing the array allocations? 
=# 

function calculate_layerThicknessEdge!(Diag::DiagnosticVars,
                                       Prog::PrognosticVars,
                                       Mesh::Mesh;
                                       backend = KA.CPU())
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack nEdges, cellsOnEdge = Edges
    @unpack nVertLevels, maxLevelEdge = VertMesh 

    # get the current timelevel of layerThickness
    layerThickness = Prog.layerThickness[end]
    # unpack the layer thickness edge diagnostic term
    @unpack layerThicknessEdge = Diag 
    
    nthreads = 100
    kernel! = compute_layerThicknessEdge!(backend, nthreads)
    # use kernel to compute diagnostic field
    kernel!(layerThicknessEdge, 
            layerThickness,
            cellsOnEdge,
            maxLevelEdge.Top,
            nEdges, nVertLevels,
            ndrange = nEdges)

    # sync the backend 
    KA.synchronize(backend)
    
    # pack the diagnostic field back into the struct for further computation
    @pack! Diag = layerThicknessEdge
end 

@kernel function compute_layerThicknessEdge!(layerThicknessEdge,
                                             @Const(layerThickness),
                                             @Const(cellsOnEdge),
                                             @Const(maxLevelEdgeTop),
                                             @Const(nEdges),
                                             @Const(nVertLevels))

    iEdge = @index(Global, Linear)

    if iEdge < nEdges + 1
        
        # initialize to avoid divide by zero and NaN problems
        @inbounds for k in 1:nVertLevels
            @inbounds layerThicknessEdge[k, iEdge] = -1.0e34
        end

        @inbounds for k in 1:maxLevelEdgeTop[iEdge]

            @inbounds @private iCell1 = cellsOnEdge[1,iEdge]
            @inbounds @private iCell2 = cellsOnEdge[2,iEdge]

            @inbounds layerThicknessEdge[k, iEdge] = 0.5 *
                (layerThickness[k, iCell1] + layerThickness[k, iCell2])
        end
    end

    @synchronize()
end

function calculate_thicknessFlux!(Diag::DiagnosticVars,
                                  Prog::PrognosticVars,
                                  Mesh::Mesh;
                                  backend = CUDABackend())

    @unpack nEdges, edgeMask = Mesh.HorzMesh.Edges

    normalVelocity = Prog.normalVelocity[end]
    @unpack thicknessFlux, layerThicknessEdge = Diag

    nthreads = 100
    kernel!  = compute_thicknessFlux!(backend, nthreads)

    kernel!(thicknessFlux,
            Prog.normalVelocity[end],
            layerThicknessEdge,
            edgeMask,
            nEdges, ndrange=nEdges)

    @pack! Diag = thicknessFlux
end

@kernel function compute_thicknessFlux!(thicknessFlux,
                                        @Const(normalVelocity),
                                        @Const(layerThicknessEdge),
                                        @Const(edgeMask),
                                        arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds thicknessFlux[1,j] = normalVelocity[1,j] *
                                       layerThicknessEdge[1,j] *
                                       edgeMask[1, j]
    end

    #k, j = @index(Global, NTuple)
    #if j < arrayLength + 1
    #    @inbounds thicknessFlux[k,j] = normalVelocity[k,j,end] * layerThicknessEdge[k,j]
    #end
    @synchronize()
end

function calculate_velocityDivCell!(Diag::DiagnosticVars,
                                    Prog::PrognosticVars,
                                    Mesh::Mesh;
                                    backend = KA.CPU()) 
    
    normalVelocity = Prog.normalVelocity[end]

    # I think the issue is that this doesn't create a new array while the old version does... we need a
    # new array for temporary data

    # layerThicknessEdge is used here to temporarily store intermdeiate results. It will be reset when it is acually
    # used as a diagnostic variable
    @unpack velocityDivCell, layerThicknessEdge = Diag


    DivergenceOnCell!(velocityDivCell, normalVelocity, layerThicknessEdge, Mesh; backend=backend)

    @pack! Diag = velocityDivCell
end

@kernel function compute_relativeVorticity!(relativeVorticity,
                                            @Const(normalVelocity),
                                            @Const(edgesOnVertex),
                                            @Const(dcEdge), 
                                            @Const(edgeSignOnVertex),
                                            @Const(areaTriangle), 
                                            @Const(vertexDegree),
                                            @Const(maxLevelVertexBot))

    # global indicies over nVertices
    iVertex = @index(Global, Linear)

    #@inbounds @private 
    invAreaTriangle = 1.0 / areaTriangle[iVertex]

    for j in 1:vertexDegree
        #@inbounds 
        iEdge = edgesOnVertex[j, iVertex]
        
        # padded iEdge array would probably be better
        if iEdge > 0 break end

        for k in 1:maxLevelVertexBot[iVertex]
            # TODO: Add support for free-slip and partial slip
            relativeVorticity[k, iVertex] += dcEdge[iEdge] *
                                             invAreaTriangle *
                                             normalVelocity[k, iEdge] *
                                             edgeSignOnVertex[j, iVertex]
        end
    end
end

function calculate_relativeVorticity!(Diag::DiagnosticVars, 
                                      Prog::PrognosticVars, 
                                      Mesh::Mesh;
                                      backend = KA.CPU()) 

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack DualCells, Edges = HorzMesh

    @unpack nEdges, dcEdge = Edges
    @unpack maxLevelVertex = VertMesh 
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, edgeSignOnVertex, edgesOnVertex = DualCells

    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity[end]
    # unpack the relativeVorticity diagnostic term
    @unpack relativeVorticity = Diag

    #nthreads = 50
    kernel!  = compute_relativeVorticity!(backend)#, nthreads)
    # use kernel to compute diagnostic field
    kernel!(relativeVorticity,
            normalVelocity,
            edgesOnVertex,
            dcEdge,
            edgeSignOnVertex, 
            areaTriangle,
            vertexDegree,
            maxLevelVertex.Bot,
            ndrange=nVertices)

    # sync the backend 
    KA.synchronize(backend)

    # pack the diagnostic field back into the struct for further computation
    @pack! Diag = relativeVorticity
end
