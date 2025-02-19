using UnPack

import Adapt
import MOKA.MPASMesh: padded_index_array
#using MPAS_O: GlobalConfig, Mesh, ConfigGet, NCDataset

mutable struct PrognosticVars{F<:AbstractFloat, FV1 <: AbstractArray{F,1}, FV2 <: AbstractArray{F,2}, VFV1 <: AbstractVector{FV1}, VFV2 <: AbstractVector{FV2}}

    # var: sea surface height [m] 
    # dim: (nCells, Time)
    ssh::VFV1

    # var: horizonal velocity, normal component to an edge [m s^{-1}]
    # dim: (nVertLevels, nEdges, Time)
    normalVelocity::VFV2

    # var: layer thickness [m]
    # dim: (nVertLevels, nCells, Time)
    layerThickness::VFV2

    ## var: potential temperature [deg C]
    ## dim: (nVertLevels, nCells, Time)
    #temperature::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)

    ## var: salinity [g salt per kg seawater]
    ## dim: (nVertLevels, nCells, Time)
    #salinity::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)
    
    function PrognosticVars(ssh::AT1D,
                            normalVelocity::AT2D,
                            layerThickness::AT2D,
                            nTimeLevels) where {AT1D, AT2D}

        # pack all the arguments into a tuple for type and backend checking
        args = (ssh, normalVelocity, layerThickness)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        #check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        # Stack args into vectors:
        sshVector = Vector{AT1D}(undef, nTimeLevels)
        normalVelocityVector = Vector{AT2D}(undef, nTimeLevels)
        layerThicknessVector = Vector{AT2D}(undef, nTimeLevels)

        for j = 1:nTimeLevels
            sshVector[j] = deepcopy(ssh)
            normalVelocityVector[j] = deepcopy(normalVelocity)
            layerThicknessVector[j] = deepcopy(layerThickness)
        end

        new{type, AT1D, AT2D, Vector{AT1D}, Vector{AT2D}}(sshVector, normalVelocityVector, layerThicknessVector)
    end      
end 

function PrognosticVars(config::GlobalConfig, mesh::Mesh; backend=KA.CPU())
    
    timeManagementConfig = ConfigGet(config.namelist, "time_management")
    do_restart = ConfigGet(timeManagementConfig, "config_do_restart")
    
    if do_restart
        ArgumentError("restart not yet supported")
    else
        inputConfig = ConfigGet(config.streams, "input")
        input_filename = ConfigGet(inputConfig, "filename_template")
    end 
    # would be usefull to have option here for prescribed field for 
    # unit testing. That way a mesh file wouldn't be needed to be 
    # created to set the spatial operators 
    
    # Read the number of desired time levels from the config file 
    timeIntegrationConfig = ConfigGet(config.namelist, "time_integration")
    nTimeLevels = ConfigGet(timeIntegrationConfig, "config_number_of_time_levels")
    
    @unpack HorzMesh, VertMesh = mesh    
    @unpack PrimaryCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertLevels = VertMesh.nVertLevels

    input = NCDataset(input_filename)

    #
    # TODO: Replace these with Vector{CuMatrix} objects or something similar
    #

    ssh = padded_index_array(nCells; eltype=Float64)
    normalVelocity = zeros(Float64, nVertLevels, nEdges)
    layerThickness = zeros(Float64, nVertLevels, nCells)

    # TO DO: check that the input file only has one time level 
    # broadcast the input value across all the time levels in the Prog struct
    ssh[1:end] .= input["ssh"][:,1]
    normalVelocity[:,:] .= input["normalVelocity"][:,:,1]
    layerThickness[:,:] .= input["layerThickness"][:,:,1]
    
    # return instance of Prognostic struct 
    PrognosticVars(Adapt.adapt(backend, ssh),
                   Adapt.adapt(backend, normalVelocity),
                   Adapt.adapt(backend, layerThickness),
                   nTimeLevels)
end

function Adapt.adapt_structure(to, x::PrognosticVars)
    return PrognosticVars(Adapt.adapt(to, x.ssh[1]),
                          Adapt.adapt(to, x.normalVelocity[1]), 
                          Adapt.adapt(to, x.layerThickness[1]),
                          length(x.ssh))
end
