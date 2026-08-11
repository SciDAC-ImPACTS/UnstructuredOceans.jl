using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using Checkpointing
using FiniteDifferences
using MOKA
using CUDA
import CUDA: @allowscalar
import MOKA.MPASMesh

# %%
function ocn_run_with_ad(config_fp, k, backend)
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    timestep  = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    Mesh = Setup.mesh

    integrator = parse_integrator(
        MOKA.config_get(MOKA.config_get(Setup.config.namelist, "time_integration"),
                       "config_time_integrator"))

    # Number of fixed-size timesteps from start until the simulation-end alarm.
    total_ms = Dates.value(Millisecond(simulationAlarm.ringTime - clock.startTime))
    dt_ms    = Dates.value(Millisecond(clock.timeStep))
    nsteps   = Int(total_ms ÷ dt_ms)

    # Bundle the differentiated state into a single OceanModel so Checkpointing.jl can snapshot it.
    model   = OceanModel(integrator, Prog, Diag, Tend, Mesh, timestep)
    d_model = OceanModel(integrator,
                         Enzyme.make_zero(Prog),
                         Enzyme.make_zero(Diag),
                         Enzyme.make_zero(Tend),
                         Mesh,
                         Enzyme.make_zero(timestep))

    ocn_run_loop_checkpointed!(model, d_model, nsteps; mode = Enzyme.Reverse, nsnaps=4, verbose=1)

    d_Prog = d_model.Prog
    @show d_Prog.normalVelocity[end][1:10]
    @show d_Prog.layerThickness[end][1:10]

    write_netcdf(Setup, Diag, Prog, d_Prog)

    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"
    println("Moka.jl ran on $arch")
    println(clock.currTime)
    return @allowscalar(d_Prog.layerThickness[end][1, k]), @allowscalar(d_Prog.normalVelocity[end][1, k])
end

function ocn_run_fd(config_fp, k, backend)
    _, _, _, Prog0 = ocn_init(config_fp, backend=backend)
    x_layer = @allowscalar Prog0.layerThickness[end][1, k]
    x_vel   = @allowscalar Prog0.normalVelocity[end][1, k]

    println("For cell number $k")

    fdm = central_fdm(5, 1)

    function run_model(config_fp, backend, perturb!)
        Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
        clock, sim_alarm, out_alarm = ocn_init_alarms(Setup)
        timestep = KA.zeros(backend, Float64, (1,))
        sumGPU   = KA.zeros(backend, Float64, (1,))
        @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
        integrator = parse_integrator(
            MOKA.config_get(MOKA.config_get(Setup.config.namelist, "time_integration"),
                           "config_time_integrator"))
        perturb!(Prog)
        ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Setup.mesh, integrator,
                     clock, sim_alarm, out_alarm)
        Array(sumGPU)[1]
    end

    f_layer = x -> run_model(config_fp, backend, Prog -> (@allowscalar Prog.layerThickness[end][1, k] = x))
    d_layer_fd = fdm(f_layer, x_layer)

    f_vel = x -> run_model(config_fp, backend, Prog -> (@allowscalar Prog.normalVelocity[end][1, k] = x))
    d_vel_fd = fdm(f_vel, x_vel)

    @show d_layer_fd
    @show d_vel_fd
    return d_layer_fd, d_vel_fd
end

# %% Run
res = "10km"
cd(joinpath(@__DIR__, res))
config_fn = "./enzyme_config.yml"

cell = 5
arch = KA.CPU()
# arch = CUDABackend()

# Enzyme reverse mode stores its per-thread tape in device-side malloc'd buffers;
# the default ~8 MB CUDA heap overflows at model scale and faults with an illegal
# memory access. Raise it before differentiating (no-op on CPU).
set_ad_device_heap!(arch)

d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, arch)
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, arch)

println("AD vs FD comparison for cell $cell")
@show (d_firstlayer_ad, d_firstlayer_fd)
@show (d_firstvelocity_ad, d_firstvelocity_fd);

# %%
@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-4)
