# Resolution-scaling benchmark — shared machinery
# ================================================
# Included by resolution_scaling_benchmark.jl (forward model) and
# resolution_scaling_ad_benchmark.jl (checkpointed reverse-mode AD). Holds
# everything both drivers share: the resolution ladders, backend/resolution
# selection, the BenchmarkTools timing harness, the per-sample state reset
# helpers, and the CSV writer. The mode-specific setup/reset/run functions and
# the device malloc-heap sizing (AD only) live in the two entry-point scripts.
#
# NB: this file is `include`d, not `using`d — it assumes the caller has already
# brought MOKA, KernelAbstractions, BenchmarkTools, Dates, DelimitedFiles and
# Printf into scope. Only the AD script additionally loads Enzyme +
# Checkpointing, so nothing here may reference them.
#
# GPU-vendor loading lives here (not in the entry scripts) so the same benchmark
# runs on either an NVIDIA or an AMD box without editing: each vendor package is
# loaded only if installed, and contributes its GPU backend only if a functional
# device is present. `@allowscalar` (used by the entry scripts) is taken from
# GPUArraysCore so it is vendor-agnostic.

# --- GPU vendors ----------------------------------------------------------------
# Detect and load whichever GPU vendor packages are installed, selecting a device
# on each. Each functional vendor contributes a "name => KA backend" entry to
# GPU_BACKENDS. Runtime comparisons are GPU-vs-GPU, so the CPU is NOT included
# here (it is available only when explicitly requested via RES_BENCH_BACKENDS=CPU;
# see selected_backends).
const GPU_BACKENDS = Pair{String,Any}[]

try
    @eval import CUDA
    if CUDA.functional()
        CUDA.device!(parse(Int, get(ENV, "RES_BENCH_CUDA_DEVICE", "1")))
        push!(GPU_BACKENDS, "CUDA" => CUDA.CUDABackend())
    end
catch err
    @debug "CUDA not available for the resolution benchmark" exception = err
end

try
    @eval import AMDGPU
    if AMDGPU.has_rocm_gpu()
        AMDGPU.device!(parse(Int, get(ENV, "RES_BENCH_AMD_DEVICE", "1")))
        push!(GPU_BACKENDS, "AMD" => AMDGPU.ROCBackend())
    end
catch err
    @debug "AMDGPU not available for the resolution benchmark" exception = err
end

isempty(GPU_BACKENDS) && @warn "No functional GPU (CUDA or AMD) detected; only \
    RES_BENCH_BACKENDS=CPU will produce results."

# --- Resolution ladders --------------------------------------------------------
# ncells is a documented hint only; the true count is read from the mesh at run
# time and is what gets written to the CSV. Each ladder is four nested resolutions
# of ONE problem (each refinement quadruples the cell count).
const GYRE = joinpath(@__DIR__, "barotropic_gyre")
const IGW  = joinpath(@__DIR__, "inertial_gravity_wave")

const LADDERS = Dict(
    "gyre" => [
        (name = "40km", dir = joinpath(GYRE, "40km"), config = "config.yml", ncells = 1_020),
        (name = "20km", dir = joinpath(GYRE, "20km"), config = "config.yml", ncells = 4_200),
        (name = "10km", dir = joinpath(GYRE, "10km"), config = "config.yml", ncells = 16_560),
        (name = "5km",  dir = joinpath(GYRE, "5km"),  config = "config.yml", ncells = 66_720),
    ],
    "igw" => [
        (name = "200km", dir = joinpath(IGW, "200km"), config = "config.yml", ncells = 2_500),
        (name = "100km", dir = joinpath(IGW, "100km"), config = "config.yml", ncells = 10_000),
        (name = "50km",  dir = joinpath(IGW, "50km"),  config = "config.yml", ncells = 40_000),
        (name = "25km",  dir = joinpath(IGW, "25km"),  config = "config.yml", ncells = 160_000),
    ],
)

const PROBLEM = get(ENV, "RES_BENCH_PROBLEM", "gyre")
haskey(LADDERS, PROBLEM) || error("RES_BENCH_PROBLEM=$PROBLEM not in $(keys(LADDERS))")
const ALL_RES = LADDERS[PROBLEM]

function selected_resolutions()
    haskey(ENV, "RES_BENCH_RES") || return ALL_RES
    want = split(ENV["RES_BENCH_RES"], ',')
    return [r for w in want for r in ALL_RES if r.name == w]
end

# Backends to sweep: name => KA backend. The default sweep is GPU-only — every
# functional GPU vendor detected above (CUDA and/or AMD) — because the runtime
# comparison is GPU-vs-GPU; the CPU is excluded from the default so it does not
# dominate the runtime plots. The CPU is still available on request via
# RES_BENCH_BACKENDS (e.g. =CUDA,CPU or =CPU) for a one-off cross-check.
function selected_backends()
    available = copy(GPU_BACKENDS)
    push!(available, "CPU" => KA.CPU())
    if !haskey(ENV, "RES_BENCH_BACKENDS")
        # Default: all detected GPUs, no CPU.
        return copy(GPU_BACKENDS)
    end
    want = split(ENV["RES_BENCH_BACKENDS"], ',')
    return filter(p -> first(p) in want, available)
end

const INTEGRATOR = parse_integrator(get(ENV, "RES_BENCH_INTEGRATOR", "ForwardEuler"))
const SAMPLES    = parse(Int, get(ENV, "RES_BENCH_SAMPLES", "5"))
const SECONDS    = parse(Float64, get(ENV, "RES_BENCH_SECONDS", "600"))

# Override the simulation-end alarm so the run stops after exactly `nsteps` steps.
# advance! increments currTime by one timeStep per iteration and the alarm rings on
# exact equality, so startTime + nsteps*timeStep lands precisely on step `nsteps`.
function set_nsteps!(clock, simulationAlarm, nsteps)
    simulationAlarm.ringTime = clock.startTime + nsteps * clock.timeStep
    simulationAlarm.ringing  = false
    simulationAlarm.stopped  = false
    return nothing
end

_dt_seconds(Setup) = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

# --- Per-sample state reset ------------------------------------------------------
# BenchmarkTools re-runs the timed loop many times, but the loop mutates its state
# (Prog advances, the clock/alarms ring, AD shadows accumulate). These helpers make
# each sample start from the identical initial state without re-paying ocn_init:
# snapshot the prognostic fields once after init, then restore them (plus clock/
# alarms/shadows) in the benchmark's `setup` phase, outside the timed region.

snapshot_prog(Prog) = (ssh            = map(copy, Prog.ssh),
                       normalVelocity = map(copy, Prog.normalVelocity),
                       layerThickness = map(copy, Prog.layerThickness))

function restore_prog!(Prog, snap)
    foreach(copyto!, Prog.ssh,            snap.ssh)
    foreach(copyto!, Prog.normalVelocity, snap.normalVelocity)
    foreach(copyto!, Prog.layerThickness, snap.layerThickness)
    return nothing
end

# Alarms are small plain mutable structs; save/restore them field-by-field so the
# periodic output alarm replays identically in every sample.
save_fields(x) = Tuple(getfield(x, f) for f in fieldnames(typeof(x)))
function restore_fields!(x, saved)
    for (f, v) in zip(fieldnames(typeof(x)), saved)
        setfield!(x, f, v)
    end
    return nothing
end

# --- BenchmarkTools timing harness -----------------------------------------------
# Times `run!` with BenchmarkTools: `reset!` runs as the per-sample setup (untimed),
# evals=1 so every evaluation starts from the freshly reset state. The warm-up call
# beforehand pays all Julia/Enzyme/CUDA compilation, so samples measure pure
# steady-state runtime. `run!` must return the scalar loss. Returns (trial, ncells,
# loss).
function benchmark_case(setup_fn, reset!, run!, dir, config, nsteps, backend)
    st = setup_fn(dir, config, nsteps, backend)
    reset!(st)
    loss = run!(st)                      # warm-up: JIT compile, untimed
    bench = @benchmarkable $run!($st) setup = ($reset!($st)) evals = 1 samples = SAMPLES seconds = SECONDS
    trial = run(bench)
    return trial, st.ncells, loss
end

seconds_stats(trial) = (min    = minimum(trial).time / 1e9,
                        median = BenchmarkTools.median(trial).time / 1e9,
                        mean   = BenchmarkTools.mean(trial).time / 1e9,
                        std    = BenchmarkTools.std(trial).time / 1e9,
                        n      = length(trial.times))

# --- Generic sweep driver --------------------------------------------------------
# Sweeps the selected backends × resolutions for one `mode`, timing each case with
# `benchmark_case`, and writes the rows to `csv_name`. `per_backend_hook(backend,
# resolutions)` runs once per backend before its cases (the AD script uses it to
# raise the CUDA malloc heap); it defaults to a no-op.
function run_sweep(; mode, nsteps, setup_fn, reset!, run!, csv_name,
                     per_backend_hook = (backend, resolutions) -> nothing)
    backends    = selected_backends()
    resolutions = selected_resolutions()

    rows = Vector{Any}[]
    for (bname, backend) in backends
        per_backend_hook(backend, resolutions)
        println("\n=== [$bname] mode=$mode  integrator=$(INTEGRATOR)  nsteps=$nsteps  samples=$SAMPLES ===")

        for res in resolutions
            print("  [$(res.name)] init + warm-up + $SAMPLES samples... "); flush(stdout)
            trial, ncells, loss = try
                benchmark_case(setup_fn, reset!, run!, res.dir, res.config, nsteps, backend)
            catch err
                println("FAILED: $(sprint(showerror, err))")
                continue
            end
            s = seconds_stats(trial)
            @printf("done\n  [%-5s] ncells=%7d  min=%9.4fs  median=%9.4fs  (%.4f s/step, n=%d)  loss=%.6e\n",
                    res.name, ncells, s.min, s.median, s.min / nsteps, s.n, loss)
            push!(rows, Any[PROBLEM, res.name, bname, mode, ncells, nsteps, s.n,
                            s.min, s.min / nsteps, s.median, s.mean, s.std, loss])
            GC.gc()
        end
    end

    # runtime_s / s_per_step are the MINIMUM over samples (the standard BenchmarkTools
    # estimator: least noise-contaminated); median/mean/std are also recorded.
    header = ["problem" "res" "backend" "mode" "ncells" "nsteps" "samples" "runtime_s" "s_per_step" "median_s" "mean_s" "std_s" "loss"]
    data   = permutedims(hcat(rows...))
    out    = joinpath(@__DIR__, csv_name)
    open(out, "w") do io
        writedlm(io, header, ',')
        writedlm(io, data, ',')
    end
    println("\nWrote $out")
    return out
end
