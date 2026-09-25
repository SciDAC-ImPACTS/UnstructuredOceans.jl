using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using Checkpointing
using FiniteDifferences
using UnstructuredOceans
using CUDA
import CUDA: @allowscalar
import UnstructuredOceans.MPASMesh

time_step_seconds(Setup) = Float64(Dates.value(Second(Setup.timeManager.timeStep)))

function configured_integrator(Setup)
    group = UnstructuredOceans.config_get(Setup.config.namelist, "time_integration")
    return parse_integrator(UnstructuredOceans.config_get(group, "config_time_integrator"))
end

function simulation_steps(clock, simulation_alarm)
    duration = Dates.value(Millisecond(simulation_alarm.ringTime - clock.startTime))
    timestep = Dates.value(Millisecond(clock.timeStep))
    return duration ÷ timestep
end

function ocn_run_with_ad(config_path, cell, backend)
    Setup, Diag, Tend, Prog = ocn_init(config_path; backend)
    clock, simulation_alarm, _ = ocn_init_alarms(Setup)
    timestep = KA.zeros(backend, Float64, 1)
    @allowscalar timestep[1] = time_step_seconds(Setup)

    integrator = configured_integrator(Setup)
    model = OceanModel(integrator, Prog, Diag, Tend, Setup.mesh, timestep)
    d_model = OceanModel(integrator,
                         Enzyme.make_zero(Prog),
                         Enzyme.make_zero(Diag),
                         Enzyme.make_zero(Tend),
                         Setup.mesh,
                         Enzyme.make_zero(timestep))

    nsteps = simulation_steps(clock, simulation_alarm)
    ocn_run_loop_checkpointed!(model, d_model, nsteps;
                                mode = Enzyme.Reverse, nsnaps = 4, verbose = 1)

    d_prog = d_model.Prog
    @show d_prog.normalVelocity[end][1:10]
    @show d_prog.layerThickness[end][1:10]
    write_netcdf(Setup, Diag, Prog, d_prog)

    architecture = backend isa KA.GPU ? "GPU" : "CPU"
    println("UnstructuredOceans.jl ran on $architecture")
    return @allowscalar(d_prog.layerThickness[end][1, cell]),
           @allowscalar(d_prog.normalVelocity[end][1, cell])
end

function run_forward(config_path, backend, perturb!)
    Setup, Diag, Tend, Prog = ocn_init(config_path; backend)
    clock, simulation_alarm, output_alarm = ocn_init_alarms(Setup)
    timestep = KA.zeros(backend, Float64, 1)
    sum_gpu = KA.zeros(backend, Float64, 1)
    @allowscalar timestep[1] = time_step_seconds(Setup)

    perturb!(Prog)
    ocn_run_loop(sum_gpu, timestep, Prog, Diag, Tend, Setup.mesh,
                 configured_integrator(Setup), clock, simulation_alarm, output_alarm)
    return Array(sum_gpu)[1]
end

function ocn_run_fd(config_path, cell, backend)
    _, _, _, initial_prog = ocn_init(config_path; backend)
    initial_layer = @allowscalar initial_prog.layerThickness[end][1, cell]
    initial_velocity = @allowscalar initial_prog.normalVelocity[end][1, cell]
    derivative = central_fdm(5, 1)

    layer_loss = value -> run_forward(config_path, backend,
        prog -> (@allowscalar prog.layerThickness[end][1, cell] = value))
    velocity_loss = value -> run_forward(config_path, backend,
        prog -> (@allowscalar prog.normalVelocity[end][1, cell] = value))

    d_layer_fd = derivative(layer_loss, initial_layer)
    d_velocity_fd = derivative(velocity_loss, initial_velocity)
    @show d_layer_fd d_velocity_fd
    return d_layer_fd, d_velocity_fd
end

# %% Run
const RESOLUTION = get(ENV, "GYRE_AD_RESOLUTION", "10km")
const CELL = parse(Int, get(ENV, "GYRE_AD_CELL", "5"))
const BACKEND = uppercase(get(ENV, "GYRE_AD_BACKEND", "CUDA")) == "CPU" ? KA.CPU() : CUDABackend()

# Set GYRE_AD_HEAP_BYTES when the selected GPU needs a larger Enzyme tape heap.
if haskey(ENV, "GYRE_AD_HEAP_BYTES")
    set_ad_device_heap!(BACKEND; bytes = parse(Int, ENV["GYRE_AD_HEAP_BYTES"]))
end

ad_result, fd_result = cd(joinpath(@__DIR__, RESOLUTION)) do
    config_path = "enzyme_config.yml"
    return ocn_run_with_ad(config_path, CELL, BACKEND), ocn_run_fd(config_path, CELL, BACKEND)
end

println("AD vs. finite differences for cell $CELL")
@show ad_result fd_result

@test isapprox(ad_result[1], fd_result[1]; atol = 1e-4)
@test isapprox(ad_result[2], fd_result[2]; atol = 1e-4)
