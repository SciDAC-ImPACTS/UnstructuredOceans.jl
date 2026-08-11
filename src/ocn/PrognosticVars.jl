using UnPack

import Adapt
#using MPAS_O: GlobalConfig, Mesh, ConfigGet, NCDataset

"""
    PrognosticVars

The state variables advanced in time by the model. Each field is a `Vector` over
time levels (typically two, the current and next step) so the integrator can hold
successive states:

- `ssh` — sea-surface height ``[\\mathrm{m}]``, dimensioned `(nCells,)` per level.
- `normalVelocity` — edge-normal horizontal velocity ``[\\mathrm{m\\,s^{-1}}]``,
  dimensioned `(nVertLevels, nEdges)` per level.
- `layerThickness` — layer thickness ``[\\mathrm{m}]``, dimensioned
  `(nVertLevels, nCells)` per level.

Construct one from a config and [`Mesh`](@ref) with
`PrognosticVars(config, mesh; backend=...)` (reads the initial state from the
input stream), or directly from arrays. Arrays live on the requested
KernelAbstractions backend; `Adapt.adapt` moves the whole struct between host and
device. See [`ocn_init`](@ref).
"""
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
    function PrognosticVars(ssh::AT1D,
                            normalVelocity::AT2D,
                            layerThickness::AT2D,
                            nTimeLevels) where {AT1D, AT2D}

        # pack all the arguments into a tuple for type and backend checking
        args = (ssh, normalVelocity, layerThickness)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
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

    ssh = zeros(Float64, nCells)
    normalVelocity = zeros(Float64, nVertLevels, nEdges)
    layerThickness = zeros(Float64, nVertLevels, nCells)

    # TO DO: check that the input file only has one time level 
    # broadcast the input value across all the time levels in the Prog struct
    ssh[:] .= input["ssh"][:,1]
    normalVelocity[:,:] .= input["normalVelocity"][:,:,1]
    layerThickness[:,:] .= input["layerThickness"][:,:,1]
    
    # return instance of Prognostic struct 
    PrognosticVars(Adapt.adapt(backend, ssh),
                   Adapt.adapt(backend, normalVelocity),
                   Adapt.adapt(backend, layerThickness),
                   nTimeLevels)
end

function Adapt.adapt_structure(to, x::PrognosticVars)
    return PrognosticVars(Adapt.adapt(to, x.ssh[end]),
                          Adapt.adapt(to, x.normalVelocity[end]),
                          Adapt.adapt(to, x.layerThickness[end]),
                          length(x.ssh))
end
