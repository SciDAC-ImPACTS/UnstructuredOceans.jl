using Test
using Dates
using CUDA: @allowscalar, CUDABackend
import KernelAbstractions as KA
using Enzyme
using FiniteDifferences
using MOKA

# Enzyme.autodiff does not support keyword arguments, so `backend` is positional here.
# The objective sum(ssh²) is computed with a direct Julia loop rather than the KA
# `sumArray` kernel: that kernel's @Const(array) annotation causes Enzyme to generate
# an incorrect backward, producing gradients that diverge from finite differences.
function ocn_run_loop_enzyme(
    sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler,
    clock, simulationAlarm, outputAlarm, backend
)
    ocn_run_loop(
        sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler,
        clock, simulationAlarm, outputAlarm; backend=backend
    )
end

# Runs forward model with AD, and computes FD derivative approximations for comparison
function ocn_run_with_ad(config_fp, k, backend)

    #
    # Setup for model
    #

    # Initialize the Model
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    timestep = KA.zeros(backend, Float64, (1,))
    d_timestep = KA.zeros(backend, Float64, (1,))
    sumGPU = KA.zeros(backend, Float64, (1,))
    d_sumGPU = KA.zeros(backend, Float64, (1,))

    sumCPU = zeros(Float64, (1,))
    d_sumCPU = zeros(Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    # Actual Model Run with AD
    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)
    d_Setup = Enzyme.make_zero(Setup)
    d_clock = Enzyme.make_zero(clock)
    d_simulationAlarm = Enzyme.make_zero(simulationAlarm)
    d_outputAlarm = Enzyme.make_zero(outputAlarm)

    autodiff(Enzyme.Reverse,
        ocn_run_loop_enzyme,
        Duplicated(sumCPU, d_sumCPU),
        Duplicated(sumGPU, d_sumGPU),
        Duplicated(timestep, d_timestep),
        Duplicated(Prog, d_Prog),
        Duplicated(Diag, d_Diag),
        Duplicated(Tend, d_Tend),
        Duplicated(Setup, d_Setup),
        Const(ForwardEuler),
        Duplicated(clock, d_clock),
        Duplicated(simulationAlarm, d_simulationAlarm),
        Duplicated(outputAlarm, d_outputAlarm),
        Const(backend)
    )

    @show d_Prog.normalVelocity[end][1:10]
    @show d_Prog.layerThickness[end][1:10]

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
    Setup0, _, _, Prog0 = ocn_init(config_fp, backend=backend)
    x_layer = @allowscalar Prog0.layerThickness[end][1, k]
    x_vel   = @allowscalar Prog0.normalVelocity[end][1, k]

    println("For cell number $k")

    fdm = central_fdm(5, 1)

    # Helper: fresh model run returning sum(ssh²) given a perturbed scalar input
    function run_model(config_fp, backend, perturb!)
        Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
        clock, sim_alarm, out_alarm = ocn_init_alarms(Setup)
        timestep = KA.zeros(backend, Float64, (1,))
        sumGPU   = KA.zeros(backend, Float64, (1,))
        sumCPU   = zeros(Float64, (1,))
        @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
        perturb!(Prog)
        ocn_run_loop(sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler,
                     clock, sim_alarm, out_alarm; backend=backend)
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
config_fn = "./config.yml"

cell = 5
d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, KA.CPU())
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, KA.CPU())

# %%
@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-4)

# %% CUDA AD test
# d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, CUDABackend())
# d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, CUDABackend())

# @test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
# @test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-2)