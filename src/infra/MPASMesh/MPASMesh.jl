module MPASMesh

# MPASMesh
export Mesh
# VertMesh.jl
export VerticalMesh
# HorzMesh.jl
export Cell, Edge, Vertex, ReadHorzMesh, HorzMesh

using UnPack
using Accessors
using NCDatasets
using KernelAbstractions 
using OffsetArrays

using MOKA

import Adapt
import KernelAbstractions as KA
import MOKA: GlobalConfig, yaml_config, ConfigGet

struct Mesh{HM,VM}
    HorzMesh::HM
    VertMesh::VM
end

function Mesh(Config::GlobalConfig; backend=KA.CPU())
    # get mesh section of the streams file
    meshConfig = ConfigGet(Config.streams, "mesh")
    # get mesh filepath from streams section
    meshPath = ConfigGet(meshConfig, "filename_template")
    # read the mesh file once
    mesh_ds = NCDataset(meshPath, "r", format=:netcdf4)
    # checks nVertLevels from config file to ensure consitency with mesh file
    nVertLevels = validate_vertical_mesh_args(mesh_ds, meshConfig)

    return Mesh(mesh_ds; nVertLevels=nVertLevels, backend=backend)
end 

function Mesh(mesh_fp::String; kwargs...)
    Mesh(NCDataset(mesh_fp, "r", format=:netcdf4); kwargs...)
end

function Mesh(mesh_ds::NCDataset; nVertLevels=nothing, backend=KA.CPU())
    # Read in the purely horizontal mesh on the CPU
    HorzMesh = ReadHorzMesh(mesh_ds)

    # Create a vertical mesh on the CPU, using the horizontal mesh	
    if isnothing(nVertLevels)
        # created a vertical mesh based on info in NetCDF file
        VertMesh = VerticalMesh(mesh_ds, HorzMesh)
    else
        # check kwarg nVertLevels is consitent with the input mesh file
        nVertLevels = validate_vertical_mesh_args(mesh_ds, nVertLevels)
        # create a stacked vertical mesh with (n) vertical levels
        VertMesh = VerticalMesh(HorzMesh; nVertLevels=nVertLevels)
    end
    
    # With both a horizontal and vertical mesh, now initalize the boundary mask
    HorzMesh = setBoundaryMask(HorzMesh, VertMesh)
    # Create the full Mesh strucutre on the CPU
    MPASMesh = Mesh(HorzMesh, VertMesh)
    
    return MPASMesh
end

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh))
end

function validate_vertical_mesh_args(mesh_ds::NCDataset, meshConfig::yaml_config)

    # TODO: Try parsing the nVertLevels from the config file
    nVertLevels = nothing

    # parse nVertLevels from the NetCDF file
    _nVertLevels = has_vertical_dim(mesh_ds) ? mesh_ds.dim["nVertLevels"] : nothing

    if all(.!isnothing.([nVertLevels,_nVertLevels]))
        if nVertLevels != _nVertLevels
            err = DimensionMismatch("")
            @error sprint(showerror, err)
            throw(err)
        end
    else
        return nVertLevels
    end
end

function validate_vertical_mesh_args(mesh_ds::NCDataset, nVertLevels::Int)
    # parse nVertLevels from the NetCDF file
    _nVertLevels = has_vertical_dim(mesh_ds) ? mesh_ds.dim["nVertLevels"] : nothing

    if all(.!isnothing.([nVertLevels,_nVertLevels]))
        if nVertLevels != _nVertLevels
            err = DimensionMismatch("")
            @error sprint(showerror, err)
            throw(err)
        end
    else
        return nVertLevels
    end
end

has_vertical_dim(mesh_ds::NCDataset) = haskey(mesh_ds.dim, "nVertLevels") 

include("HorzMesh.jl")
include("VertMesh.jl")

end 
