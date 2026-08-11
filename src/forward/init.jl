using MOKA.MPASMesh

"""
    ocn_init(Config_filepath; backend=KernelAbstractions.CPU())
        -> (Setup, Diag, Tend, Prog)

Initialize the ocean model from a YAML configuration file.

Reads the configuration at `Config_filepath`, builds the horizontal and vertical
[`Mesh`](@ref) and the simulation [`Clock`](@ref), and allocates the prognostic,
diagnostic, and tendency state on `backend`.

`backend` is a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend (e.g. `KernelAbstractions.CPU()`, or a GPU backend obtained from
[`device`](@ref) of a [`GPU`](@ref) architecture). All model arrays are allocated
on that backend.

Returns a 4-tuple `(Setup, Diag, Tend, Prog)`:
- `Setup` — a `ModelSetup` bundling the parsed config, `Mesh`, and `Clock`.
- `Diag` — [`DiagnosticVars`](@ref) (edge thickness, fluxes, divergence, vorticity).
- `Tend` — tendency accumulators for the prognostic variables.
- `Prog` — [`PrognosticVars`](@ref) initialized from the input stream.

The returned `Setup` carries the alarms needed to drive the run; recover them with
[`ocn_init_alarms`](@ref) and advance the model with [`ocn_run_loop`](@ref).
"""
function ocn_init(Config_filepath; backend=KA.CPU())
    
    # read the configuration file 
    Config = config_read(Config_filepath)
    
    #TO DO: Read constants ?? 

    # setup the mesh 
    Mesh = ocn_setup_mesh(Config; backend=backend)
    # setup clock 
    Clock = ocn_setup_clock(Config)
    
    # return the model setup instance
    Setup = ModelSetup(Config, Mesh, Clock)

    # Prognostics should be intialized here, 
    # add option to read from input file (i.e. mesh) or from restrart
    Prog = PrognosticVars(Config, Mesh; backend=backend)
    
    # Diagnostic and Tendecies probabl don't need to be initialized here 
    # instead should happen within the `ocn_run` method, prior to entering 
    # the first time integration loop to ensure values are initialized 
    Diag = DiagnosticVars(Mesh; backend=backend)
    Tend = TendencyVars(Mesh; backend=backend)

    return Setup, Diag, Tend, Prog
end 

"""
    ocn_init_shadows(Prog, Diag, Tend; backend=KernelAbstractions.CPU())
        -> d_Prog

Allocate a zero-initialized shadow (derivative) copy of the prognostic state for
use as an Enzyme differential argument (`Duplicated`).

The returned `d_Prog` has the same shape as `Prog` with all fields set to zero,
ready to be seeded by, or to receive, sensitivities during automatic
differentiation. See the [Automatic differentiation](@ref) guide.
"""
function ocn_init_shadows(Prog, Diag, Tend; backend=KA.CPU())

    d_Prog = PrognosticVars(zeros(Float64, size(Prog.ssh)),
                            zeros(Float64, size(Prog.normalVelocity)),
                            zeros(Float64, size(Prog.layerThickness)),
                            1)

    return d_Prog
end


# Default physical constants; overridable via the config `constants` section.
const DEFAULT_GRAVITY = 9.80616   # [m s^-2]
const DEFAULT_DENSITY = 1000.0    # [kg m^-3]

"""
    ocn_build_constants(Config) -> Constants

Read the optional `constants` namelist section and build the physical
[`Constants`](@ref) (gravity, density), falling back to `DEFAULT_GRAVITY` /
`DEFAULT_DENSITY` when the section or a key is absent.
"""
function ocn_build_constants(Config::GlobalConfig)
    gravity = DEFAULT_GRAVITY
    density = DEFAULT_DENSITY
    if config_has(Config.namelist, "constants")
        c = config_get(Config.namelist, "constants")
        gravity = Float64(config_get(c, "config_gravity", DEFAULT_GRAVITY))
        density = Float64(config_get(c, "config_density", DEFAULT_DENSITY))
    end
    return (gravity=gravity, density=density)
end

function ocn_setup_mesh(Config::GlobalConfig; backend=KA.CPU())
    meshConfig = config_get(Config.streams, "mesh")
    mesh_fp = config_get(meshConfig, "filename_template")

    # host scalars from config, then the device-array Constants struct.
    consts    = ocn_build_constants(Config)
    density   = consts.density   # scales wind forcing on the edges (was 1000.0)
    constants = Constants(; gravity=consts.gravity, density=density, backend=backend)

    # read del2 momentum viscosity; default 0 (no mixing) if section absent or the
    # term is switched off via config_use_mom_del2.
    momentumDel2 = 0.0
    if config_has(Config.namelist, "hmix_del2")
        hmixConfig = config_get(Config.namelist, "hmix_del2")
        if config_get(hmixConfig, "config_use_mom_del2", true) &&
           config_has(hmixConfig, "config_mom_del2")
            momentumDel2 = Float64(config_get(hmixConfig, "config_mom_del2"))
        end
    end

    h_mesh = read_horz_mesh(mesh_fp; backend=backend, momentumDel2=momentumDel2)
    v_mesh = VerticalMesh(mesh_fp, h_mesh; backend=backend)

    # wind-stress forcing gate: config_use_bulk_wind_stress in the forcing section
    # (default on for backwards compatibility with configs that omit it).
    use_wind_stress = true
    if config_has(Config.namelist, "forcing")
        forcingConfig = config_get(Config.namelist, "forcing")
        use_wind_stress = Bool(config_get(forcingConfig, "config_use_bulk_wind_stress", true))
    end

    # forcing stream file (falls back to the mesh file, which is where the BG wind
    # fields live). Build the ForcingVars (wind stress projected onto edge normals).
    forcing_fp = mesh_fp
    if config_has(Config.streams, "forcing")
        forcingStream = config_get(Config.streams, "forcing")
        forcing_fp = config_get(forcingStream, "filename_template", mesh_fp)
    end
    forcing = MOKA.MPASMesh.build_forcing(h_mesh, forcing_fp;
                                          rho=density, use_wind_stress=use_wind_stress,
                                          backend=backend)

    return Mesh(h_mesh, v_mesh, constants, forcing)
end

function ocn_setup_clock(Config::GlobalConfig)

    # Get the nested Config objects 
    outputConfig = config_get(Config.streams, "output")
    time_managementConfig = config_get(Config.namelist, "time_management")
    time_integrationConfig = config_get(Config.namelist, "time_integration")
    
    dt = config_get(time_integrationConfig, "config_dt")
    stop_time = config_get(time_managementConfig, "config_stop_time")
    start_time = config_get(time_managementConfig, "config_start_time")
    run_duration = config_get(time_managementConfig, "config_run_duration")
    restart_timestamp_name = config_get(time_managementConfig, "config_restart_timestamp_name")
    
    output_reference_time = config_get(outputConfig, "reference_time")
    output_interval = config_get(outputConfig, "output_interval")

    if run_duration != "none" 
        clock = mpas_create_clock(dt, start_time; runDuration=run_duration)
        if stop_time != "none" 
            start_time + run_duration != stop_time && println("Warning: config_run_duration and config_stop_time are inconsitent: using config_run_duration.")
        else 
            stop_time = start_time + run_duration
        end 
    elseif stop_time != "none"
        clock = mpas_create_clock(dt, start_time; stopTime=stop_time)
    else 
        throw("Error: Neither config_run_duration nor config_stop_time were specified.")
    end
    
    # create the end of simulation alarm 
    simulationAlarm = OneTimeAlarm("simulation_end", stop_time)
    # attached the simulation_end alarm to the clock 
    attach_alarm!(clock, simulationAlarm)

    # create the ouput alarm
    outputAlarm = PeriodicAlarm("outputAlarm", output_interval, output_reference_time)
    # attach the output alarm to the clock 
    attach_alarm!(clock, outputAlarm)

    return clock
end 

"""
    ocn_init_alarms(Setup) -> (clock, simulationAlarm, outputAlarm)

Retrieve the simulation clock and its two driving alarms from an initialized
`Setup`.

`ocn_init` attaches a one-time `"simulation_end"` alarm and a periodic
`"outputAlarm"` to the clock; this helper unpacks them for passing to
[`ocn_run_loop`](@ref). Returns the [`Clock`](@ref), the end-of-run
[`OneTimeAlarm`](@ref), and the output [`PeriodicAlarm`](@ref).
"""
function ocn_init_alarms(Setup)
    clock = Setup.timeManager

    simulationAlarm = clock.alarms["simulation_end"]
    outputAlarm = clock.alarms["outputAlarm"]

    return clock, simulationAlarm, outputAlarm
end