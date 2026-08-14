# %%
using Dates
using MOKA
using Statistics
import KernelAbstractions as KA
using GPUArraysCore: @allowscalar

function ocn_run(config_fp, arch::AbstractArchitecture = CPU())
    backend = MOKA.device(arch)
    println("Setting the backend...")
    @show backend

    # Initialize the model
    Setup, Diag, Tend, Prog = ocn_init(config_fp; backend=backend)
    println("Initialized the model")
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    println("Initialized the clock.")
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    ti_str = MOKA.ConfigGet(MOKA.ConfigGet(Setup.config.namelist, "time_integration"), "config_time_integrator")
    integrator = parse_integrator(ti_str)
    println("Time integrator: $integrator")
    output_ds = io_initialize(Setup, Prog)
    ocn_run_loop(timestep, Prog, Diag, Tend, Setup, integrator, clock, simulationAlarm, outputAlarm; output_ds=output_ds)
    io_finalize(output_ds)

    arch_str = arch isa GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch_str")
    println(clock.currTime)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        if length(ARGS) == 2 && ARGS[2] == "cuda"
            using CUDA
            ocn_run(ARGS[1], GPU())
        else
            ocn_run(ARGS[1])
        end
    else
        error("yaml config file invalid")
    end
end
