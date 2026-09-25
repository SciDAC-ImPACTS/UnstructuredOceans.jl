# Whole-model AD-vs-finite-difference gradient regression test.
#
# Exercises the full checkpointed reverse-mode adjoint end to end — every tendency
# kernel (divergence, curl, coriolis, thicknessFluxDiv, horizontal momentum mixing)
# in the real model — and checks the gradient of loss = Σ ssh[end]^2 with respect to
# an initial normalVelocity entry against central finite differences. This is the
# regression guard for the tendency kernels' reverse-mode correctness (the operator
# tests cover gradient/divergence/curl in isolation).

using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using Checkpointing
using UnstructuredOceans
using CUDA
using CUDA: @allowscalar

const MODEL_NSTEPS = 4
const MODEL_NSNAPS = 4
# Smallest gyre case (1020 cells) keeps the adjoint compile/run tractable.
const CASE_DIR = normpath(joinpath(@__DIR__, "..", "..",
                                   "examples", "barotropic_gyre", "40km"))

reset_clock!(c, a, nsteps) = (c.currTime = c.startTime; c.prevTime = nothing;
    c.nextTime = c.startTime + c.timeStep; a.ringTime = c.startTime + nsteps * c.timeStep;
    a.ringing = false; a.stopped = false; nothing)
save_model_fields(x) = Tuple(getfield(x, f) for f in fieldnames(typeof(x)))
restore_model_fields!(x, v) = (for (f, val) in zip(fieldnames(typeof(x)), v); setfield!(x, f, val); end; nothing)
snap_prog(P) = (; ssh = map(copy, P.ssh), normalVelocity = map(copy, P.normalVelocity),
                  layerThickness = map(copy, P.layerThickness))
function restore_prog!(P, s)
    foreach(copyto!, P.ssh, s.ssh)
    foreach(copyto!, P.normalVelocity, s.normalVelocity)
    foreach(copyto!, P.layerThickness, s.layerThickness)
    return nothing
end

@testset "whole-model checkpointed adjoint vs finite differences" begin
    backends = CUDA.functional() ? [KA.CPU(), CUDABackend()] : [KA.CPU()]
    for backend in backends
        @info "whole-model gradient" backend
        set_ad_device_heap!(backend; bytes = 2 * 1024^3)

        Setup, Diag, Tend, Prog = cd(CASE_DIR) do
            ocn_init("config.yml"; backend)
        end
        clock, sim_alarm, out_alarm = ocn_init_alarms(Setup)
        reset_clock!(clock, sim_alarm, MODEL_NSTEPS)
        timestep = KA.zeros(backend, Float64, 1)
        sumbuf   = KA.zeros(backend, Float64, 1)
        @allowscalar timestep[1] = Float64(Dates.value(Second(Setup.timeManager.timeStep)))
        ti = UnstructuredOceans.config_get(Setup.config.namelist, "time_integration")
        integrator = parse_integrator(UnstructuredOceans.config_get(ti, "config_time_integrator"))
        Mesh = Setup.mesh
        prog_snap  = snap_prog(Prog)
        alarm_snap = save_model_fields(out_alarm)

        model   = OceanModel(integrator, Prog, Diag, Tend, Mesh, timestep)
        d_model = OceanModel(integrator, Enzyme.make_zero(Prog), Enzyme.make_zero(Diag),
                             Enzyme.make_zero(Tend), Mesh, Enzyme.make_zero(timestep))

        function primal_loss!()
            restore_prog!(Prog, prog_snap)
            reset_clock!(clock, sim_alarm, MODEL_NSTEPS)
            restore_model_fields!(out_alarm, alarm_snap)
            fill!(sumbuf, 0.0)
            ocn_run_loop(sumbuf, timestep, Prog, Diag, Tend, Mesh, integrator,
                         clock, sim_alarm, out_alarm)
            KA.synchronize(backend)
            return @allowscalar sumbuf[1]
        end

        # --- AD gradient of Σ ssh[end]^2 w.r.t. initial normalVelocity ---
        restore_prog!(model.Prog, prog_snap)
        Enzyme.make_zero!(d_model.Prog); Enzyme.make_zero!(d_model.Diag)
        Enzyme.make_zero!(d_model.Tend); Enzyme.make_zero!(d_model.dt)
        KA.synchronize(backend)
        ocn_run_loop_checkpointed!(model, d_model, MODEL_NSTEPS;
                                   mode = Enzyme.Reverse, nsnaps = MODEL_NSNAPS, verbose = 0)
        KA.synchronize(backend)
        gcpu = Array(d_model.Prog.normalVelocity[end])
        e0 = argmax(abs.(vec(gcpu)))          # a definitely-nonzero gradient entry
        ad = vec(gcpu)[e0]

        # --- central finite difference at the same entry ---
        nv0 = @allowscalar vec(prog_snap.normalVelocity[end])[e0]
        ϵ = 1e-4 * max(abs(nv0), 1.0)
        setnv!(v) = @allowscalar (vec(prog_snap.normalVelocity[end])[e0] = v)
        lp = (setnv!(nv0 + ϵ); primal_loss!())
        lm = (setnv!(nv0 - ϵ); primal_loss!())
        setnv!(nv0)                            # restore initial state
        fd = (lp - lm) / (2ϵ)

        @info "whole-model gradient result" backend edge=e0 ad fd
        @test abs(fd) > 1e-10                  # non-degenerate comparison
        @test isapprox(ad, fd; rtol = 1e-4, atol = 1e-8)
    end
end
