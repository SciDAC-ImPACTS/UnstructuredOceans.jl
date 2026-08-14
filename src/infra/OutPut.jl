"""
    write_netcdf(Setup, Diag, Prog, d_Prog)
    write_netcdf(Setup, Diag, Prog)

Write the model state to a NetCDF file in one shot (mesh coordinates plus the
prognostic and diagnostic fields).

The four-argument method additionally writes the AD shadow/sensitivity fields
carried in `d_Prog` (as `d_*` variables), for visualizing adjoints. State on a
GPU backend is copied back to the host before writing. For time-stepped output
during a run, use [`io_initialize`](@ref) / [`io_write_timestep`](@ref) /
[`io_finalize`](@ref) instead.
"""
function write_netcdf(Setup::ModelSetup,
                      Diag::DiagnosticVars,
                      Prog::PrognosticVars,
                      d_Prog::PrognosticVars)

    # copy the data structures back to the CPU
    Mesh = Adapt.adapt_structure(KA.CPU(), Setup.mesh)
    Diag = Adapt.adapt_structure(KA.CPU(), Diag)
    Prog = Adapt.adapt_structure(KA.CPU(), Prog)
    d_Prog = Adapt.adapt_structure(KA.CPU(), d_Prog)

    clock = Setup.timeManager
    config = Setup.config

    outputConfig = config_get(config.streams, "output")
    output_filename = config_get(outputConfig, "filename_template")
    
    # create the netCDF dataset
    ds = NCDataset(output_filename,"c")
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels
    maxEdges = PrimaryCells.maxEdges
    TWO = 2

    # hardcode everything for now out of convenience
    defDim(ds,"time",1)
    defDim(ds,"nCells", nCells)
    defDim(ds,"nEdges", nEdges)
    defDim(ds,"nVertices", nVertices)
    defDim(ds,"nVertLevels", nVertLevels)
    defDim(ds,"maxEdges", maxEdges)
    defDim(ds,"TWO", TWO)
   
    dt = convert(Float64,Second(clock.timeStep).value) 
    # define timestep as global attribute
    ds.attrib["dt"] = dt
    
    #units_string = "seconds since $(Dates.format(clock.startTime, "yyyy-mm-dd HH:MM:SS"))"
    
    # Define the coordinate variables 
    xtime = defVar(ds,"time", Float64,("time",)) #attrib = [ "units" => units_string,
                                                #           "calendar" => "julian"])
    xCell = defVar(ds,"xCell",Float64,("nCells",))
    yCell = defVar(ds,"yCell",Float64,("nCells",))
    xEdge = defVar(ds,"xEdge",Float64,("nEdges",))
    yEdge = defVar(ds,"yEdge",Float64,("nEdges",))
    xVertex = defVar(ds,"xVertex",Float64,("nVertices",))
    yVertex = defVar(ds,"yVertex",Float64,("nVertices",))
    
    # Define the mesh metric variables 
    dcEdge = defVar(ds,"dcEdge",Float64,("nEdges",))
    areaCell = defVar(ds,"areaCell",Float64,("nCells",))
    angleEdge = defVar(ds,"angleEdge",Float64,("nEdges",))
    areaTriangle = defVar(ds,"areaTriangle",Float64,("nVertices",))
    
    # Define the mesh connectivity variables 
    edgeSignOnCell = defVar(ds,"edgeSignOnCell",Int32,("maxEdges","nCells"))
    nEdgesOnCell = defVar(ds,"nEdgesOnCell",Int32,("nCells",))
    nEdgesOnEdge = defVar(ds,"nEdgesOnEdge",Int32,("nEdges",))
    cellsOnEdge = defVar(ds,"cellsOnEdge",Int32,("TWO","nEdges"))
    verticesOnCell = defVar(ds,"verticesOnCell",Int32,("maxEdges","nCells"))
    verticesOnEdge = defVar(ds,"verticesOnEdge",Int32,("TWO","nEdges"))
    
    # Define the data variables 
    ssh = defVar(ds,"ssh",Float64,("nCells","time"))
    layerThickness = defVar(ds,"layerThickness",Float64,("nCells","nVertLevels","time"))
    normalVelocity = defVar(ds,"normalVelocity",Float64,("nEdges","nVertLevels","time"))

    # Define the shadoe arrays of the data variables we're interested in
    d_ssh = defVar(ds,"d_ssh",Float64,("nCells","time"))
    d_layerThickness = defVar(ds,"d_layerThickness",Float64,("nCells","nVertLevels","time"))
    d_normalVelocity = defVar(ds,"d_normalVelocity",Float64,("nEdges","nVertLevels","time"))
    
    # dump the variables into the dataset. 
    xtime[:] = Dates.value(Second(clock.currTime - clock.startTime))
    xCell[:] = PrimaryCells.xᶜ
    yCell[:] = PrimaryCells.yᶜ
    xEdge[:] = Edges.xᵉ
    yEdge[:] = Edges.yᵉ
    xVertex[:] = DualCells.xᵛ
    yVertex[:] = DualCells.yᵛ
    
    dcEdge[:] = Edges.dcEdge
    areaCell[:] = PrimaryCells.areaCell
    #angleEdge[:] = mesh.angleEdge
    areaTriangle[:] = DualCells.areaTriangle
    
    #edgeSignOnCell[:] = mesh.HorzMesh.PrimaryCells.ESoC
    nEdgesOnCell[:] = PrimaryCells.nEdgesOnCell
    nEdgesOnEdge[:] = Edges.nEdgesOnEdge
    #cellsOnEdge[:] = mesh.HorzMesh.Edges.CoE
    #verticesOnCell[:,:] = mesh.HorzMesh.PrimaryCells.VoC
    #verticesOnEdge[:,:] = mesh.HorzMesh.Edges.VoE
    
    #@show Prog.ssh[end]
    #@show d_Prog.ssh[end]

    #@show typeof(Prog.ssh[end]), typeof(d_Prog.ssh[end])

    ssh[:,:] = Prog.ssh[end]
    layerThickness[:,:,:] = Prog.layerThickness[end] 
    normalVelocity[:,:,:] = Prog.normalVelocity[end]

    d_ssh[:,:] = d_Prog.ssh[end]
    d_layerThickness[:,:,:] = d_Prog.layerThickness[end] 
    d_normalVelocity[:,:,:] = d_Prog.normalVelocity[end]

    close(ds)
end

function write_netcdf(Setup::ModelSetup,
                      Diag::DiagnosticVars,
                      Prog::PrognosticVars)

    # copy the data structures back to the CPU
    Mesh = Adapt.adapt_structure(KA.CPU(), Setup.mesh)
    Diag = Adapt.adapt_structure(KA.CPU(), Diag)
    Prog = Adapt.adapt_structure(KA.CPU(), Prog)

    clock = Setup.timeManager
    config = Setup.config

    outputConfig = config_get(config.streams, "output")
    output_filename = config_get(outputConfig, "filename_template")

    # create the netCDF dataset
    ds = NCDataset(output_filename,"c")

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels
    maxEdges = PrimaryCells.maxEdges
    TWO = 2

    # hardcode everything for now out of convenience
    defDim(ds,"time",1)
    defDim(ds,"nCells", nCells)
    defDim(ds,"nEdges", nEdges)
    defDim(ds,"nVertices", nVertices)
    defDim(ds,"nVertLevels", nVertLevels)
    defDim(ds,"maxEdges", maxEdges)
    defDim(ds,"TWO", TWO)

    dt = convert(Float64,Second(clock.timeStep).value) 
    # define timestep as global attribute
    ds.attrib["dt"] = dt

    #units_string = "seconds since $(Dates.format(clock.startTime, "yyyy-mm-dd HH:MM:SS"))"

    # Define the coordinate variables 
    xtime = defVar(ds,"time", Float64,("time",)) #attrib = [ "units" => units_string,
                                #           "calendar" => "julian"])
    xCell = defVar(ds,"xCell",Float64,("nCells",))
    yCell = defVar(ds,"yCell",Float64,("nCells",))
    xEdge = defVar(ds,"xEdge",Float64,("nEdges",))
    yEdge = defVar(ds,"yEdge",Float64,("nEdges",))
    xVertex = defVar(ds,"xVertex",Float64,("nVertices",))
    yVertex = defVar(ds,"yVertex",Float64,("nVertices",))

    # Define the mesh metric variables 
    dcEdge = defVar(ds,"dcEdge",Float64,("nEdges",))
    areaCell = defVar(ds,"areaCell",Float64,("nCells",))
    angleEdge = defVar(ds,"angleEdge",Float64,("nEdges",))
    areaTriangle = defVar(ds,"areaTriangle",Float64,("nVertices",))

    # Define the mesh connectivity variables 
    edgeSignOnCell = defVar(ds,"edgeSignOnCell",Int32,("maxEdges","nCells"))
    nEdgesOnCell = defVar(ds,"nEdgesOnCell",Int32,("nCells",))
    nEdgesOnEdge = defVar(ds,"nEdgesOnEdge",Int32,("nEdges",))
    cellsOnEdge = defVar(ds,"cellsOnEdge",Int32,("TWO","nEdges"))
    verticesOnCell = defVar(ds,"verticesOnCell",Int32,("maxEdges","nCells"))
    verticesOnEdge = defVar(ds,"verticesOnEdge",Int32,("TWO","nEdges"))

    # Define the data variables 
    ssh = defVar(ds,"ssh",Float64,("nCells",))
    layerThickness = defVar(ds,"layerThickness",Float64,("nCells","nVertLevels"))
    normalVelocity = defVar(ds,"normalVelocity",Float64,("nEdges","nVertLevels"))

    # dump the variables into the dataset. 
    xtime[:] = Dates.value(Second(clock.currTime - clock.startTime))
    xCell[:] = PrimaryCells.xᶜ
    yCell[:] = PrimaryCells.yᶜ
    xEdge[:] = Edges.xᵉ
    yEdge[:] = Edges.yᵉ
    xVertex[:] = DualCells.xᵛ
    yVertex[:] = DualCells.yᵛ

    dcEdge[:] = Edges.dcEdge
    areaCell[:] = PrimaryCells.areaCell
    #angleEdge[:] = mesh.angleEdge
    areaTriangle[:] = DualCells.areaTriangle

    #edgeSignOnCell[:] = mesh.HorzMesh.PrimaryCells.ESoC
    nEdgesOnCell[:] = PrimaryCells.nEdgesOnCell
    nEdgesOnEdge[:] = Edges.nEdgesOnEdge
    #cellsOnEdge[:] = mesh.HorzMesh.Edges.CoE
    #verticesOnCell[:,:] = mesh.HorzMesh.PrimaryCells.VoC
    #verticesOnEdge[:,:] = mesh.HorzMesh.Edges.VoE

    ssh[:] = Prog.ssh[end]
    layerThickness[:,:] = Prog.layerThickness[end]
    normalVelocity[:,:] = Prog.normalVelocity[end]

    close(ds)
end

"""
    io_initialize(Setup, Prog) -> NCDataset

Create the output NetCDF file with an unlimited time dimension, write all static
mesh variables once, and append the initial state (t = 0) as frame 1.
Returns the open dataset so subsequent frames can be streamed in with
[`io_write_timestep`](@ref) and closed with [`io_finalize`](@ref).
"""
function io_initialize(Setup::ModelSetup,
                       Prog::PrognosticVars)

    Mesh = Adapt.adapt_structure(KA.CPU(), Setup.mesh)
    Prog = Adapt.adapt_structure(KA.CPU(), Prog)

    clock  = Setup.timeManager
    config = Setup.config

    outputConfig    = config_get(config.streams, "output")
    output_filename = config_get(outputConfig, "filename_template")

    ds = NCDataset(output_filename, "c")

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges     = Edges.nEdges
    nCells     = PrimaryCells.nCells
    nVertices  = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels
    maxEdges   = PrimaryCells.maxEdges
    TWO        = 2

    defDim(ds, "time",       Inf)   # unlimited — grows as frames are appended
    defDim(ds, "nCells",     nCells)
    defDim(ds, "nEdges",     nEdges)
    defDim(ds, "nVertices",  nVertices)
    defDim(ds, "nVertLevels", nVertLevels)
    defDim(ds, "maxEdges",   maxEdges)
    defDim(ds, "TWO",        TWO)

    dt = convert(Float64, Second(clock.timeStep).value)
    ds.attrib["dt"] = dt

    # Coordinate variables
    defVar(ds, "time",    Float64, ("time",))
    defVar(ds, "xCell",   Float64, ("nCells",))
    defVar(ds, "yCell",   Float64, ("nCells",))
    defVar(ds, "xEdge",   Float64, ("nEdges",))
    defVar(ds, "yEdge",   Float64, ("nEdges",))
    defVar(ds, "xVertex", Float64, ("nVertices",))
    defVar(ds, "yVertex", Float64, ("nVertices",))

    # Mesh metric variables
    defVar(ds, "dcEdge",      Float64, ("nEdges",))
    defVar(ds, "areaCell",    Float64, ("nCells",))
    defVar(ds, "angleEdge",   Float64, ("nEdges",))
    defVar(ds, "areaTriangle", Float64, ("nVertices",))

    # Mesh connectivity variables
    defVar(ds, "edgeSignOnCell", Int32, ("maxEdges", "nCells"))
    defVar(ds, "nEdgesOnCell",   Int32, ("nCells",))
    defVar(ds, "nEdgesOnEdge",   Int32, ("nEdges",))
    defVar(ds, "cellsOnEdge",    Int32, ("TWO", "nEdges"))
    defVar(ds, "verticesOnCell", Int32, ("maxEdges", "nCells"))
    defVar(ds, "verticesOnEdge", Int32, ("TWO", "nEdges"))

    # Data variables — trailing `time` dimension for time series
    defVar(ds, "ssh",            Float64, ("nCells",   "time"))
    defVar(ds, "layerThickness", Float64, ("nCells",   "nVertLevels", "time"))
    defVar(ds, "normalVelocity", Float64, ("nEdges",   "nVertLevels", "time"))

    # Write static mesh data
    ds["xCell"][:]   = PrimaryCells.xᶜ
    ds["yCell"][:]   = PrimaryCells.yᶜ
    ds["xEdge"][:]   = Edges.xᵉ
    ds["yEdge"][:]   = Edges.yᵉ
    ds["xVertex"][:] = DualCells.xᵛ
    ds["yVertex"][:] = DualCells.yᵛ

    ds["dcEdge"][:]      = Edges.dcEdge
    ds["areaCell"][:]    = PrimaryCells.areaCell
    ds["areaTriangle"][:] = DualCells.areaTriangle
    ds["nEdgesOnCell"][:] = PrimaryCells.nEdgesOnCell
    ds["nEdgesOnEdge"][:] = Edges.nEdgesOnEdge

    # Write initial state as frame 1 (t = 0)
    ds["time"][1]              = 0.0
    ds["ssh"][:, 1]            = Prog.ssh[end]
    ds["layerThickness"][:, :, 1] = Prog.layerThickness[end]
    ds["normalVelocity"][:, :, 1] = Prog.normalVelocity[end]

    return ds
end

"""
    io_write_timestep(ds, Setup, Prog, frame)

Append one snapshot of the prognostic fields at the current clock time to the
open dataset `ds` at the given (1-based) `frame` index.
"""
function io_write_timestep(ds::NCDataset,
                          Setup::ModelSetup,
                          Prog::PrognosticVars,
                          frame::Int)

    Prog  = Adapt.adapt_structure(KA.CPU(), Prog)
    clock = Setup.timeManager

    t = Float64(Dates.value(Second(clock.currTime - clock.startTime)))

    ds["time"][frame]              = t
    ds["ssh"][:, frame]            = Prog.ssh[end]
    ds["layerThickness"][:, :, frame] = Prog.layerThickness[end]
    ds["normalVelocity"][:, :, frame] = Prog.normalVelocity[end]

    return nothing
end

"""
    io_finalize(ds)

Close the output dataset.
"""
function io_finalize(ds::NCDataset)
    close(ds)
    return nothing
end

