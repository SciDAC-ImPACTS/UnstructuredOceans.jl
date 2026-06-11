# %%
using Dates
using CUDA
using MOKA
using Statistics
using KernelAbstractions

using CUDA: @allowscalar

# Might need to remove these:
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.ModelSetup}) = true
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.Clock}) = true

const KA=KernelAbstractions

# file path to the config file. Should be parsed from the command line 
#config_fp = "../../TestData/inertial_gravity_wave_100km.yml"
#config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"
config_fp = "./config.yml"

function ocn_run(config_fp, backend = KA.CPU())

    #
    # Setup for model
    #
    
    println("Setting the backend...")
    #backend = KA.CPU()
    # backend = CUDABackend()
    @show backend
    
    # Initialize the Model  
    Setup, Diag, Tend, Prog             = ocn_init(config_fp; backend = backend)
    println("Initialized the model")
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    println("Initialized the clock.")
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=backend)

    #
    # Writing to outputs
    #
    
    # Only suport i/o at the end of the simulation for now 
    write_netcdf(Setup, Diag, Prog)
    
    backend = get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch")
    println(clock.currTime)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        ocn_run(ARGS[1])
    else 
        error("yaml config file invalid")
    end
end
