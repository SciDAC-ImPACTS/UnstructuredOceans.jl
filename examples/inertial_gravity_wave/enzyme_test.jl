using Test
using Dates
using CUDA: @allowscalar, CUDABackend
import KernelAbstractions as KA
using Enzyme
using FiniteDifferences
using MOKA

# Runs forward model with AD, and computes FD derivative approximations for comparison
function ocn_run_with_ad(config_fp, k, backend)

    #
    # Setup for model
    #

    # Initialize the Model on the SAME backend as timestep/sumGPU below. Without
    # backend=backend the model state defaults to CPU while timestep is on GPU;
    # the Forward-Euler CPU kernel then reads timestep[1] (a GPU scalar), lowering
    # to cuMemcpyDtoHAsync_v2 with a gc-transition bundle Enzyme cannot differentiate.
    Setup, Diag, Tend, Prog = ocn_init(config_fp; backend=backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    timestep = KA.zeros(backend, Float64, (1,))
    d_timestep = KA.zeros(backend, Float64, (1,))
    sumGPU = KA.zeros(backend, Float64, (1,))
    d_sumGPU = KA.ones(backend, Float64, (1,))   # seed: d(loss)/d(sumGPU[1]) = 1.0
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    # Extract Mesh before autodiff: keeps ModelSetup (which contains yaml_config,
    # a mutable struct with Dict fields) out of the differentiated scope entirely,
    # preventing EnzymeRuntimeActivityError from the heap-pointer store analysis.
    Mesh = Setup.mesh

    # Actual Model Run with AD
    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)

    # Checkpointed adjoint: differentiates one timestep at a time so Enzyme's
    # per-step CUDA malloc-heap tape is freed between steps, instead of one
    # autodiff over the whole loop (which overflows the heap past ~4 steps).
    ocn_run_loop_checkpointed!(sumGPU, d_sumGPU, timestep, d_timestep,
                               Prog, d_Prog, Diag, d_Diag, Tend, d_Tend,
                               Mesh, ForwardEuler,
                               clock, simulationAlarm, outputAlarm)

    @show d_Prog.normalVelocity[end][1:10]
    @show d_Prog.layerThickness[end][1:10]
    # NB: d_Tend / d_Diag are intermediate shadows that the checkpointed driver
    # zeroes at the start of every reverse step, so they hold no meaningful value
    # here — the differentiated quantities of interest are the d_Prog fields.

    #
    # Writing to outputs
    #

    # Only support i/o at the end of the simulation for now
    write_netcdf(Setup, Diag, Prog, d_Prog)

    backend = KA.get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch")
    println(clock.currTime)
    return @allowscalar d_Prog.layerThickness[end][1, k], d_Prog.normalVelocity[end][1, k]
end

# Runs forward model with FiniteDifferences.jl for AD comparison
function ocn_run_fd(config_fp, k, backend)

    # Sample initial values from an unperturbed model initialisation
    Setup0, _, _, Prog0 = ocn_init(config_fp; backend=backend)
    x_layer = @allowscalar Prog0.layerThickness[end][1, k]
    x_vel   = @allowscalar Prog0.normalVelocity[end][1, k]

    println("For cell number $k")

    fdm = central_fdm(5, 1)

    # Helper: fresh model run returning sum(ssh²) given a perturbed scalar input.
    # D2H copy happens here, outside Enzyme's traced region.
    function run_model(config_fp, backend, perturb!)
        Setup, Diag, Tend, Prog = ocn_init(config_fp; backend=backend)
        clock, sim_alarm, out_alarm = ocn_init_alarms(Setup)
        timestep = KA.zeros(backend, Float64, (1,))
        sumGPU   = KA.zeros(backend, Float64, (1,))
        @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
        perturb!(Prog)
        ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Setup.mesh, ForwardEuler,
                     clock, sim_alarm, out_alarm;)
        Array(sumGPU)[1]
    end

    # FD derivative w.r.t. layerThickness[end][1, k]
    f_layer = x -> run_model(config_fp, backend, Prog -> (@allowscalar Prog.layerThickness[end][1, k] = x))
    d_layer_fd = fdm(f_layer, x_layer)

    # FD derivative w.r.t. normalVelocity[end][1, k]
    f_vel = x -> run_model(config_fp, backend, Prog -> (@allowscalar Prog.normalVelocity[end][1, k] = x))
    d_vel_fd = fdm(f_vel, x_vel)

    @show d_layer_fd
    @show d_vel_fd
    return d_layer_fd, d_vel_fd
end

# %% Replace these with inertialgravity waves and a config file:
res = "200km"
cd(joinpath(@__DIR__, res))
config_fn = "./config_enzyme.yml"

cell = 5
# arch = KA.CPU()
arch = CUDABackend()

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