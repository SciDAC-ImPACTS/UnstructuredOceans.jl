# Resolution-scaling benchmark — checkpointed reverse-mode AD
# ==========================================================
# Measures how MOKA's checkpointed reverse-mode AD driver
# (`ocn_run_loop_checkpointed!`, Revolve) scales with mesh resolution (problem
# size) on the available GPU(s) — NVIDIA (CUDA) and/or AMD (ROCm), whichever are
# installed and functional. The runtime comparison is GPU-vs-GPU, so the CPU is
# excluded by default (still runnable on request via RES_BENCH_BACKENDS=CPU). The
# plain forward model is timed by the companion script
# resolution_scaling_benchmark.jl (same ladders, same harness); the two were
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
# For a controlled comparison every resolution is differentiated over the SAME
# number of timesteps, with the model's native dt, via the checkpointed
# `ocn_run_loop_checkpointed!` (exactly as in the per-example enzyme_test.jl
# harnesses). Timing is done with BenchmarkTools so only steady-state, JIT-compiled
# runtime is measured: each case is initialized ONCE (mesh I/O and Julia/Enzyme/CUDA
# compilation stay outside the measurement), then the driver runs for several
# samples, with a cheap state reset (restore initial prognostic fields, zero the AD
# shadows) as the per-sample `setup` so every sample computes the identical adjoint.
# The CSV reports the minimum across samples (the standard low-noise estimator) plus
# median/mean/std.
#
# Results are written to resolution_scaling_ad_benchmark.csv, which
# plot_resolution_scaling_benchmark.jl turns into figures (alongside the forward CSV).
#
# Run with:  julia --project=. examples/resolution_scaling_ad_benchmark.jl
#
# Environment overrides (all optional):
#   RES_BENCH_PROBLEM=gyre        # or =igw (default: gyre)
#   RES_BENCH_BACKENDS=CUDA,AMD   # subset of CUDA,AMD,CPU (default: all detected GPUs)
#   RES_BENCH_CUDA_DEVICE=1       # CUDA device index (default 1)
#   RES_BENCH_AMD_DEVICE=1        # AMD/ROCm device index (default 1)
#   RES_BENCH_RES=40km,20km       # subset/order of resolutions to sweep
#   RES_BENCH_AD_NSTEPS=4         # AD steps per resolution (default 4)
#   RES_BENCH_INTEGRATOR=RK4      # or ForwardEuler (default ForwardEuler)
#   RES_BENCH_SAMPLES=5           # timed samples per case (default 5)
#   RES_BENCH_SECONDS=600         # BenchmarkTools time budget per case (default 600)

using Dates
import KernelAbstractions as KA
using BenchmarkTools
using Enzyme
using Checkpointing   # with Enzyme, loads MOKACheckpointingExt (ocn_run_loop_checkpointed!)
using MOKA
using GPUArraysCore: @allowscalar
using DelimitedFiles
using Printf

# resolution_scaling_common.jl loads whichever GPU vendor packages (CUDA, AMDGPU)
# are installed + functional, and picks a device on each. @allowscalar above is
# GPUArraysCore's vendor-agnostic version, so it works for CuArray and ROCArray.
include(joinpath(@__DIR__, "resolution_scaling_common.jl"))

const AD_NSTEPS = parse(Int, get(ENV, "RES_BENCH_AD_NSTEPS", "4"))

# Enzyme's reverse tape lives in the CUDA in-kernel malloc heap and scales with the
# mesh (it scatters over nEdges × nEdgesOnEdge — see set_ad_device_heap! docstring)
# AND with the number of tendency evaluations Enzyme records per differentiated step.
# When the heap is too small, malloc returns NULL and the kernel faults with an
# ERROR_ILLEGAL_ADDRESS. Size it from the cell count, keeping a per-integrator floor
# for small meshes and a safety factor for headroom. (32 GB card ⇒ 10+ GB heaps are fine.)
#
# The 512 MB / 40k-cell floor was calibrated for ForwardEuler, which evaluates the
# tendencies ONCE per step. RK4 evaluates them FOUR times per step (4 substages), so
# its tape is ~4× larger and needs a 4× floor. Empirically, at a 4 GB heap the 40k-cell
# (50 km) RK4 adjoint succeeds but the 160k-cell (25 km) one overflows; the linear
# upper bound tape(160k)=4·tape(40k) puts the 25 km requirement at ≤16 GB, which the
# RK4 floor below delivers (`ad_heap_bytes(160_000, RungeKutta4) == 16 GB`).
const _AD_HEAP_FLOOR    = 512 * 1024 * 1024  # bytes; ForwardEuler tape for ~40k cells
const _AD_HEAP_REFCELLS = 40_000             # cell count the floor is calibrated at
const _AD_HEAP_SAFETY   = 2                  # headroom multiplier above the floor

# Tendency evaluations Enzyme records per differentiated step, per integrator. This is
# the multiplier on the reverse tape (and hence the heap) relative to ForwardEuler.
_ad_stage_factor(::Type{ForwardEuler}) = 1
_ad_stage_factor(::Type{RungeKutta4})  = 4

ad_heap_bytes(ncells, integrator) =
    let floor = _AD_HEAP_FLOOR * _ad_stage_factor(integrator)
        max(floor, round(Int, _AD_HEAP_SAFETY * floor * ncells / _AD_HEAP_REFCELLS))
    end

# --- AD case: init once, then (reset!, run!) per sample --------------------------
function setup_ad(dir, config, nsteps, backend)
    cd(dir)
    Setup, Diag, Tend, Prog = ocn_init(config; backend=backend)

    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = _dt_seconds(Setup)

    Mesh = Setup.mesh

    # NB: the CUDA malloc heap must already be sized for this mesh — it is raised
    # once per backend up front (raise_ad_heap!), because cuCtxSetLimit(MALLOC_HEAP_SIZE)
    # fails with ERROR_INVALID_VALUE once the context's heap is in use (after ocn_init).

    # Bundle the differentiated state so Checkpointing.jl can snapshot it; the
    # shadow zeroes Prog/Diag/Tend/dt and aliases the (inactive) Mesh.
    model   = OceanModel(INTEGRATOR, Prog, Diag, Tend, Mesh, timestep)
    d_model = OceanModel(INTEGRATOR,
                         Enzyme.make_zero(Prog),
                         Enzyme.make_zero(Diag),
                         Enzyme.make_zero(Tend),
                         Mesh,
                         Enzyme.make_zero(timestep))

    # Revolve needs 1 <= nsnaps <= nsteps; a handful is plenty for these horizons.
    nsnaps = clamp(4, 1, nsteps)

    return (; model, d_model, nsteps, nsnaps, backend,
              snap = snapshot_prog(Prog), ncells = length(Prog.ssh[end]))
end

# Restore the primal initial state and zero the accumulated shadows. Diag/Tend
# primals need no reset: every timestep fully recomputes them from Prog before use.
# d_model.Mesh aliases the primal mesh (inactive), so it must NOT be zeroed.
function reset_ad!(st)
    restore_prog!(st.model.Prog, st.snap)
    Enzyme.make_zero!(st.d_model.Prog)
    Enzyme.make_zero!(st.d_model.Diag)
    Enzyme.make_zero!(st.d_model.Tend)
    Enzyme.make_zero!(st.d_model.dt)
    KA.synchronize(st.backend)
    return nothing
end

# Timed body: ONLY the checkpointed reverse-mode AD driver. Returns the loss.
function run_ad!(st)
    loss = ocn_run_loop_checkpointed!(st.model, st.d_model, st.nsteps;
                                      nsnaps = st.nsnaps, verbose = 0)
    KA.synchronize(st.backend)
    return loss
end

# Raise the CUDA malloc heap for AD ONCE per backend, before any ocn_init touches
# the context — cuCtxSetLimit(MALLOC_HEAP_SIZE) fails once the heap is in use. Size
# it for the LARGEST resolution we will differentiate (using the ncells hints, which
# need no init) so every mesh's Enzyme tape fits. No-op on CPU.
function raise_ad_heap!(backend, resolutions)
    max_ncells = maximum(r.ncells for r in resolutions)
    set_ad_device_heap!(backend; bytes = ad_heap_bytes(max_ncells, INTEGRATOR))
    return nothing
end

run_sweep(; mode = "ad", nsteps = AD_NSTEPS,
            setup_fn = setup_ad, reset! = reset_ad!, run! = run_ad!,
            csv_name = "resolution_scaling_ad_benchmark.csv",
            per_backend_hook = raise_ad_heap!)
