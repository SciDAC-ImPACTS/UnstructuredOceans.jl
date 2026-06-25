using Test
using Dates
using CUDA: @allowscalar, CUDABackend
import KernelAbstractions as KA
using Enzyme
using FiniteDifferences
using MOKA

# The Enzyme `inactive_type` guards for mesh / clock / alarm types now ship with
# MOKA via the MOKAEnzymeExt package extension, which loads automatically here
# because both MOKA and Enzyme are in scope. No need to declare them per-script.

# Enzyme.autodiff does not support keyword arguments, so `backend` is positional here.
# sumCPU is intentionally absent: see ocn_run_loop in run_loop.jl for the reason.
# function ocn_run_loop_enzyme(
#     sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
#     clock, simulationAlarm, outputAlarm, backend
# )
#     ocn_run_loop(
#         sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
#         clock, simulationAlarm, outputAlarm;
#     )
# end

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

    autodiff(Enzyme.Reverse,
        ocn_run_loop,
        Duplicated(sumGPU, d_sumGPU),
        Duplicated(timestep, d_timestep),
        Duplicated(Prog, d_Prog),
        Duplicated(Diag, d_Diag),
        Duplicated(Tend, d_Tend),
        Const(Mesh),
        Const(ForwardEuler),
        Const(clock),
        Const(simulationAlarm),
        Const(outputAlarm),
    )

    @show d_Prog.normalVelocity[end][1:10]
    @show d_Prog.layerThickness[end][1:10]
    # tendNormalVelocity / tendLayerThickness are 2D arrays (nVertLevels, n…),
    # not time-level vectors like the Prog fields — slice the first level rather
    # than indexing [end] (which would scalar-index the last element on the GPU).
    @show d_Tend.tendNormalVelocity[1, 1:10]
    @show d_Tend.tendLayerThickness[1, 1:10]

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
# Dedicated short-duration config for the gradient check: reverse-mode AD stores
# a tape for every kernel of every timestep, so the differentiated horizon is
# kept to a couple of steps. The full-length run uses ./config.yml (regenerated
# from ../Moka.yaml by convergence.sh).
config_fn = "./config_enzyme.yml"

cell = 5
# arch = KA.CPU()
arch = CUDABackend()

# Enzyme reverse mode stores its per-thread tape in device-side malloc'd buffers;
# the default ~8 MB CUDA heap overflows at model scale and faults with an illegal
# memory access. Raise it before differentiating (no-op on CPU).
# set_ad_device_heap!(arch)

d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, arch)
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, arch)

println("AD vs FD comparison for cell $cell")
@show (d_firstlayer_ad, d_firstlayer_fd)
@show (d_firstvelocity_ad, d_firstvelocity_fd);

# %%
@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-4)