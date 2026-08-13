using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using UnstructuredOceans
using CUDA
import CUDA: @allowscalar
CUDA.device!(1)

# Forward-mode AD of the barotropic gyre: differentiate the final model state
# with respect to the del2 momentum viscosity (config_mom_del2).
#
# Forward mode is a natural fit here: the independent variable is a single scalar
# (viscosity), so one forward sweep yields the full sensitivity of every output
# state variable. Unlike reverse mode it keeps no tape, so there is no CUDA
# in-kernel malloc-heap pressure and no need for checkpointing.
#
# viscosity is threaded into ocn_timestep as a length-1 device array `viscDel2`
# (see ocn_run_loop_fwd!). We make it a Duplicated argument with tangent 1.0; the
# shadow of Prog then holds d(final state)/d(viscosity).

function _nsteps(Setup)
    clock, simAlarm, _ = ocn_init_alarms(Setup)
    total_ms = Dates.value(Millisecond(simAlarm.ringTime - clock.startTime))
    dt_ms    = Dates.value(Millisecond(clock.timeStep))
    return Int(total_ms ÷ dt_ms)
end

function ocn_run_with_fwd_ad(config_fp, k, backend)
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    Mesh = Setup.mesh
    integrator = parse_integrator(
        UnstructuredOceans.config_get(UnstructuredOceans.config_get(Setup.config.namelist, "time_integration"),
                       "config_time_integrator"))

    nsteps = _nsteps(Setup)
    println("Running $nsteps forward-mode steps")

    # Viscosity as an explicit differentiated input, seeded with tangent 1.
    visc0    = @allowscalar Mesh.HorzMesh.Edges.momentumDel2[1]
    viscDel2 = KA.zeros(backend, Float64, (1,))
    @allowscalar viscDel2[1] = visc0
    d_viscDel2 = KA.zeros(backend, Float64, (1,))
    @allowscalar d_viscDel2[1] = 1.0

    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)

    autodiff(Forward, ocn_run_loop_fwd!,
             Duplicated(viscDel2, d_viscDel2),
             Const(timestep),
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Const(Mesh),
             Const(integrator),
             Const(nsteps))

    d_layer = @allowscalar d_Prog.layerThickness[end][1, k]
    d_vel   = @allowscalar d_Prog.normalVelocity[end][1, k]
    @show d_layer d_vel
    return d_layer, d_vel
end

# Central finite difference w.r.t. viscosity, with a FIXED step h. We do not use
# FiniteDifferences.central_fdm here: its adaptive step selection shrinks h toward
# the scale that is optimal for O(1) outputs, but the viscosity sensitivities here
# are O(1e-10), so that tiny step lands in floating-point roundoff and the estimate
# is dominated by noise. A manual step of h=10 (≈2.5% of the 400 m²/s base value)
# is safely in the truncation-limited regime — a convergence sweep (h = 40…0.1)
# shows the estimate is flat to ~1e-5 relative for h in [2.5, 40] and only degrades
# once h drops below ~1 (see fd_convergence_check.jl).
function ocn_run_fd(config_fp, k, backend; h = 10.0)
    println("For cell number $k")

    function run_model(config_fp, backend, visc)
        Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
        timestep = KA.zeros(backend, Float64, (1,))
        @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
        Mesh = Setup.mesh
        integrator = parse_integrator(
            UnstructuredOceans.config_get(UnstructuredOceans.config_get(Setup.config.namelist, "time_integration"),
                           "config_time_integrator"))
        nsteps = _nsteps(Setup)
        viscDel2 = KA.zeros(backend, Float64, (1,))
        @allowscalar viscDel2[1] = visc
        ocn_run_loop_fwd!(viscDel2, timestep, Prog, Diag, Tend, Mesh, integrator, nsteps)
        return (@allowscalar(Prog.layerThickness[end][1, k]),
                @allowscalar(Prog.normalVelocity[end][1, k]))
    end

    Setup, _, _, _ = ocn_init(config_fp, backend=backend)
    visc0 = @allowscalar Setup.mesh.HorzMesh.Edges.momentumDel2[1]

    lp, vp = run_model(config_fp, backend, visc0 + h)
    lm, vm = run_model(config_fp, backend, visc0 - h)
    d_layer_fd = (lp - lm) / (2h)
    d_vel_fd   = (vp - vm) / (2h)
    @show d_layer_fd d_vel_fd
    return d_layer_fd, d_vel_fd
end

# %% Run
res = "10km"
cd(joinpath(@__DIR__, res))
# Longer spin-up (1 h at dt=40 s → 90 RK4 steps) than enzyme_config_short.yml: over
# 20 s viscosity leaves no imprint above finite-difference roundoff noise (the FD
# derivative is ~1e-11 numerical dust), so a meaningful AD-vs-FD check needs enough
# steps for the del2 mixing term to actually shape the flow.
config_fn = "./enzyme_config_fwd.yml"

cell = 5
arch = CUDABackend()

println("=== Forward-mode AD ===")
d_layer_ad, d_vel_ad = ocn_run_with_fwd_ad(config_fn, cell, arch)
println("=== Finite differences ===")
d_layer_fd, d_vel_fd = ocn_run_fd(config_fn, cell, arch)

println("AD vs FD comparison (d(final state)/d(viscosity)) for cell $cell")
@show (d_layer_ad, d_layer_fd)
@show (d_vel_ad, d_vel_fd)

# %%
# AD is exact; the residual is finite-difference truncation/roundoff. The
# convergence sweep (fd_convergence_check.jl) puts the fixed-step FD within ~1e-4
# relative of AD, so 1e-3 rtol is a comfortable, non-flaky bound.
@test isapprox(d_layer_ad, d_layer_fd, rtol=1e-3, atol=1e-14)
@test isapprox(d_vel_ad,   d_vel_fd,   rtol=1e-3, atol=1e-14)
