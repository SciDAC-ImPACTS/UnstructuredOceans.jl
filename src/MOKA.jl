module MOKA

    export ocn_run_loop, ocn_run_loop_fwd!, ocn_run_loop_checkpointed!, ocn_loss, OceanModel, ocn_step!, ocn_init, ocn_init_shadows, ocn_init_alarms, is_ringing, advance!, ocn_timestep, change_time_step!, reset!
    export RungeKutta4, ForwardEuler, parse_integrator
    export write_netcdf, io_initialize, io_write_timestep, io_finalize
    export Clock, OneTimeAlarm, PeriodicAlarm

    # MPASMesh
    export VerticalMesh, read_horz_mesh, Mesh, HorzMesh,
           Cell, Edge, Vertex

    # Operators
    export GradientOnEdge!,
           DivergenceOnCell!,
           CurlOnVertex!,
           ZeroOutVector!

    # Architectures
    export CPU, GPU, AbstractArchitecture, AbstractSerialArchitecture,
           device, architecture, array_type, on_architecture,
           set_ad_device_heap!, DEFAULT_NTHREADS

    using Dates, Printf, YAML, NCDatasets, UnPack, Statistics, Logging, KernelAbstractions

    include("Architectures.jl")

    # include infrastructure code
    # (Should all of this just be it's own module which is imported here?)
    include("infra/Config.jl")
    include("infra/TimeManager.jl")
    include("infra/MPASMesh/MPASMesh.jl")
    include("infra/ModelSetup.jl")


    include("ocn/Operators.jl")
    include("ocn/PrognosticVars.jl")
    include("ocn/DiagnosticVars.jl")

    # This infrastructure code is lower down b/c it depends on Prog/Diag structures
    # for now, so those have to be defined before it can be included
    include("infra/OutPut.jl")

    include("ocn/Tendencies/TendencyVars.jl")
    include("ocn/Tendencies/normalVelocity/normalVelocity.jl")
    include("ocn/Tendencies/layerThickness/layerThickness.jl")

    include("forward/init.jl")
    include("forward/time_integration.jl")
    include("forward/run_loop.jl")

    ###
    ### Needed so we can export names from sub-modules at the top level
    ###
    using .MPASMesh
    using .NormalVelocity
    using .LayerThickness
end
