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

    outputConfig = ConfigGet(config.streams, "output")
    output_filename = ConfigGet(outputConfig, "filename_template")
    
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

    ssh[:,:] = Prog.ssh[end][1:end]
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

    outputConfig = ConfigGet(config.streams, "output")
    output_filename = ConfigGet(outputConfig, "filename_template")

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

    ssh[:] = Prog.ssh[end][1:end]
    layerThickness[:,:] = Prog.layerThickness[end]
    normalVelocity[:,:] = Prog.normalVelocity[end]

    close(ds)
end

#function io_writeTimestep()
#end 
#
#function io_finalize(ds)
#end

