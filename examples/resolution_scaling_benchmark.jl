# Resolution-scaling benchmark — forward model
# ============================================
# Measures how MOKA's forward-model runtime scales with mesh resolution (problem
# size) on the available GPU(s) — NVIDIA (CUDA) and/or AMD (ROCm), whichever are
# installed and functional. The runtime comparison is GPU-vs-GPU, so the CPU is
# excluded by default (still runnable on request via RES_BENCH_BACKENDS=CPU). The
# checkpointed reverse-mode AD driver is timed by the companion script
# resolution_scaling_ad_benchmark.jl (same ladders, same harness); the two were
# split so a forward-only sweep needn't load Enzyme / Checkpointing and so each
# can be run and tuned independently.
#
# Two problem families ship nested resolutions of the SAME problem, so each is a
# natural scaling ladder — every refinement quadruples the cell count while the
# physics is unchanged. Pick one with RES_BENCH_PROBLEM (default: gyre):
#
#   barotropic_gyre (gyre):        inertial_gravity_wave (igw):
#      40 km ->    1,020 cells        200 km ->    2,500 cells
#      20 km ->    4,200 cells   (x4) 100 km ->   10,000 cells   (x4)
#      10 km ->   16,560 cells   (x4)  50 km ->   40,000 cells   (x4)
#       5 km ->   66,720 cells   (x4)  25 km ->  160,000 cells   (x4)
#
# (The gyre 40 km / 20 km meshes under-resolve the Munk boundary layer, so they
# are not physically converged — but this benchmark times compute cost per step,
# which is a function of mesh size alone, so that is immaterial here.)
#
# For a controlled comparison every resolution is advanced the SAME number of
# timesteps (compute per step is what scales with the mesh), with the model's
# native dt, using the I/O-free `ocn_run_loop`. Timing is done with BenchmarkTools
# so only steady-state, JIT-compiled runtime is measured: each case is initialized
# ONCE (mesh I/O and compilation stay outside the measurement), then the timed loop
# runs for several samples, with a cheap state reset (restore initial prognostic
# fields, rewind the clock/alarms) as the per-sample `setup`. The CSV reports the
# minimum across samples (the standard low-noise estimator) plus median/mean/std.
#
# Results are written to resolution_scaling_benchmark.csv, which
# plot_resolution_scaling_benchmark.jl turns into figures (alongside the AD CSV).
#
# Run with:  julia --project=. examples/resolution_scaling_benchmark.jl
#
# Environment overrides (all optional):
#   RES_BENCH_PROBLEM=gyre        # or =igw (default: gyre)
#   RES_BENCH_BACKENDS=CUDA,AMD   # subset of CUDA,AMD,CPU (default: all detected GPUs)
#   RES_BENCH_CUDA_DEVICE=1       # CUDA device index (default 1)
#   RES_BENCH_AMD_DEVICE=1        # AMD/ROCm device index (default 1)
#   RES_BENCH_RES=40km,20km       # subset/order of resolutions to sweep
#   RES_BENCH_FWD_NSTEPS=8        # forward steps per resolution (default 8)
#   RES_BENCH_INTEGRATOR=RK4      # or ForwardEuler (default ForwardEuler)
#   RES_BENCH_SAMPLES=5           # timed samples per case (default 5)
#   RES_BENCH_SECONDS=600         # BenchmarkTools time budget per case (default 600)

using Dates
import KernelAbstractions as KA
using BenchmarkTools
using MOKA
using GPUArraysCore: @allowscalar
using DelimitedFiles
using Printf

# resolution_scaling_common.jl loads whichever GPU vendor packages (CUDA, AMDGPU)
# are installed + functional, and picks a device on each. @allowscalar above is
# GPUArraysCore's vendor-agnostic version, so it works for CuArray and ROCArray.
include(joinpath(@__DIR__, "resolution_scaling_common.jl"))

const FWD_NSTEPS = parse(Int, get(ENV, "RES_BENCH_FWD_NSTEPS", "8"))

# --- Forward case: init once, then (reset!, run!) per sample ---------------------
function setup_forward(dir, config, nsteps, backend)
    cd(dir)
    Setup, Diag, Tend, Prog = ocn_init(config; backend=backend)
    clock, simAlarm, outAlarm = ocn_init_alarms(Setup)
    set_nsteps!(clock, simAlarm, nsteps)

    timestep = KA.zeros(backend, Float64, (1,))
    sumGPU   = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = _dt_seconds(Setup)

    return (; Prog, Diag, Tend, Mesh = Setup.mesh,
              clock, simAlarm, outAlarm, timestep, sumGPU, nsteps, backend,
              snap = snapshot_prog(Prog), outAlarmSaved = save_fields(outAlarm),
              ncells = length(Prog.ssh[end]))
end

function reset_forward!(st)
    restore_prog!(st.Prog, st.snap)
    st.clock.currTime = st.clock.startTime
    st.clock.prevTime = nothing
    st.clock.nextTime = st.clock.startTime + st.clock.timeStep
    set_nsteps!(st.clock, st.simAlarm, st.nsteps)
    restore_fields!(st.outAlarm, st.outAlarmSaved)
    fill!(st.sumGPU, 0.0)
    KA.synchronize(st.backend)
    return nothing
end

# Timed body: ONLY the forward loop (init / mesh I/O excluded). Uses the I/O-free
# `ocn_run_loop` variant so we measure pure compute. That variant accumulates the
# loss sumGPU[1] = Σ ssh[end]²; read it back (outside the timed region) so every CSV
# row carries a finite loss.
function run_forward!(st)
    ocn_run_loop(st.sumGPU, st.timestep, st.Prog, st.Diag, st.Tend, st.Mesh,
                 INTEGRATOR, st.clock, st.simAlarm, st.outAlarm)
    KA.synchronize(st.backend)
    return @allowscalar st.sumGPU[1]
end

run_sweep(; mode = "forward", nsteps = FWD_NSTEPS,
            setup_fn = setup_forward, reset! = reset_forward!, run! = run_forward!,
            csv_name = "resolution_scaling_benchmark.csv")
