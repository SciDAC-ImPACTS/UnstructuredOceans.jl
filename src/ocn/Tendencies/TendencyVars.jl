#abstract type StateVar end 
#normalVelocity <: StateVar
#layerThickness <: StateVar
#temperature <: StateVar
#salinty <: StateVar

mutable struct TendencyVars{F<:AbstractFloat, FV2 <: AbstractArray{F,2}}
    
    # var: time tendency of normal component of velociy [m s^{-2}]
    # dim: (nVertLevels, nEdges), Time?)
    tendNormalVelocity::FV2

    # var: time tendency of layer thickness [m s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    tendLayerThickness::FV2
    
    #= UNUSED FOR NOW:
    # var: time tendency of potential temperature [\deg C s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    temperatureTend::FV2

    # var: time tendency of salinity measured as change in practical 
    #      salinity unit per second [PSU s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    salinityTend::FV2

    # NOTE: I don't think this needs to be a Tendency term. 
    # var: time tendency of sea-surface height [m s^{-1}]
    # dim: (nCells), Time?)
    tendSSH::Array{F,1}
    =#

    function TendencyVars(tendNormalVelocity::AT2D,
                          tendLayerThickness::AT2D) where {AT2D}

        # pack all the arguments into a tuple for type and backend checking
        args = (tendNormalVelocity, tendLayerThickness)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(tendNormalVelocity, tendLayerThickness)
    end
end 

function TendencyVars(Mesh::Mesh; backend=CUDABackend())
        
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertLevels = VertMesh.nVertLevels

    # create zero vectors to store tendecy vars on the desired backend
    tendNormalVelocity = zeros(Float64, nVertLevels, nEdges) 
    tendLayerThickness = KA.zeros(backend, Float64, nVertLevels, nCells)



    TendencyVars(Adapt.adapt(backend, tendNormalVelocity), tendLayerThickness)
end 

function axb!(a::Array{T,2}, x::T, b::Array{T,2}) where {T<:AbstractFloat}
    m,n = size(a)

    @boundscheck (m,n) == size(b) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
           a[i,j] += x*b[i,j]
        end
    end
end 

