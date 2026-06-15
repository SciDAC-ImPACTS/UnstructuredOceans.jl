using Test
using Dates
using CUDA: @allowscalar, CUDABackend
import KernelAbstractions as KA
using Enzyme
using FiniteDifferences
using MOKA

# KA Kernel objects (backend + function ref) are not differentiable; marking them
# inactive forces Enzyme to treat kernel locals as Const, which is required for
# KA's own augmented_primal/reverse rules (func::Const{<:Kernel}) to fire instead
# of Enzyme tracing into CUDA kernel compilation code that involves Module types.
Enzyme.EnzymeRules.inactive_type(::Type{<:KA.Kernel}) = true

# Mesh geometry is never differentiable. These inactive_type declarations cover
# the full type hierarchy so Enzyme's static analysis never treats mesh-derived
# CuArrays as potentially-active. Without this, two failure modes occur:
#   1. EnzymeRuntimeActivityError: Enzyme sees a Const mutable struct's CuArray
#      heap pointer stored to the callee's stack (!enzymejl_byref_MUT_REF).
#   2. GPU segfault at synchronize: runtime activity analysis incorrectly activates
#      mesh CuArrays that are annotated @Const in GPU kernels; reverse kernels then
#      try to write gradients to read-only GPU memory.
import MOKA.MPASMesh
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Mesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.HorzMesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.VerticalMesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.PrimaryCells}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.DualCells}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Edges}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.ActiveLevels}) = true

# Clock and alarm types contain DateTime / Dict / Period fields that are not
# differentiable.  Enzyme.make_zero produces invalid shadows for them and can
# corrupt the backward tape, triggering GPU segfaults.  Mark the whole type
# hierarchy inactive; clock/alarms are pure loop-control, not model state.
Enzyme.EnzymeRules.inactive_type(::Type{<:Clock}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:OneTimeAlarm}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:PeriodicAlarm}) = true

# Enzyme.autodiff does not support keyword arguments, so `backend` is positional here.
# sumCPU is intentionally absent: see ocn_run_loop in run_loop.jl for the reason.
function ocn_run_loop_enzyme(
    sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
    clock, simulationAlarm, outputAlarm, backend
)
    ocn_run_loop(
        sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
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
        ocn_run_loop_enzyme,
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

    # Helper: fresh model run returning sum(ssh²) given a perturbed scalar input.
    # D2H copy happens here, outside Enzyme's traced region.
    function run_model(config_fp, backend, perturb!)
        Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
        clock, sim_alarm, out_alarm = ocn_init_alarms(Setup)
        timestep = KA.zeros(backend, Float64, (1,))
        sumGPU   = KA.zeros(backend, Float64, (1,))
        @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
        perturb!(Prog)
        ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Setup.mesh, ForwardEuler,
                     clock, sim_alarm, out_alarm; backend=backend)
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
config_fn = "./config.yml"

cell = 5
# arch = KA.CPU()
arch = CUDABackend()
d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, arch)
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, arch)

# %%
@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-4)

# %% CUDA AD test
# d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, CUDABackend())
# d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, CUDABackend())

# @test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
# @test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-2)