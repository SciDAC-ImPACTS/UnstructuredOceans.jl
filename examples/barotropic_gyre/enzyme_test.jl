using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using FiniteDifferences
using MOKA

Enzyme.EnzymeRules.inactive_type(::Type{<:KA.Kernel}) = true

import MOKA.MPASMesh
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Mesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.HorzMesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.VerticalMesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.PrimaryCells}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.DualCells}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Edges}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.ActiveLevels}) = true

Enzyme.EnzymeRules.inactive_type(::Type{<:Clock}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:OneTimeAlarm}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:PeriodicAlarm}) = true

function ocn_run_loop_enzyme(
    sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
    clock, simulationAlarm, outputAlarm, backend
)
    ocn_run_loop(
        sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler,
        clock, simulationAlarm, outputAlarm; backend=backend
    )
end

function ocn_run_with_ad(config_fp, k, backend)
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    timestep  = KA.zeros(backend, Float64, (1,))
    d_timestep = KA.zeros(backend, Float64, (1,))
    sumGPU    = KA.zeros(backend, Float64, (1,))
    d_sumGPU  = KA.ones(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    Mesh = Setup.mesh

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
        perturb!(Prog)
        ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Setup.mesh, ForwardEuler,
                     clock, sim_alarm, out_alarm; backend=backend)
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

res = "10km_old"
cd(joinpath(@__DIR__, res))
config_fn = "./enzyme_config.yml"

cell = 5
arch = KA.CPU()

d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, arch)
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, arch)

@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-4)
