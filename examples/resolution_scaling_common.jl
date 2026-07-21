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
#
# DEVICE_LABELS records, per backend name, the concrete hardware string (GPU model
# / CPU name). It is written into every CSV row (alongside the host node) so that
# results appended from many HPC compute nodes — each possibly a different GPU —
# stay attributable; see the append-mode CSV writer in run_sweep.
const GPU_BACKENDS  = Pair{String,Any}[]
const DEVICE_LABELS = Dict{String,String}("CPU" => Sys.CPU_NAME)

try
    @eval import CUDA
    if CUDA.functional()
        CUDA.device!(parse(Int, get(ENV, "RES_BENCH_CUDA_DEVICE", "0")))
        push!(GPU_BACKENDS, "CUDA" => CUDA.CUDABackend())
        DEVICE_LABELS["CUDA"] = CUDA.name(CUDA.device())
    end
catch err
    @debug "CUDA not available for the resolution benchmark" exception = err
end

try
    @eval import AMDGPU
    if AMDGPU.has_rocm_gpu()
        AMDGPU.device_id!(parse(Int, get(ENV, "RES_BENCH_AMD_DEVICE", "1")))
        push!(GPU_BACKENDS, "AMD" => AMDGPU.ROCBackend())
        DEVICE_LABELS["AMD"] = AMDGPU.HIP.name(AMDGPU.device())
    end
catch err
    @debug "AMDGPU not available for the resolution benchmark" exception = err
end

isempty(GPU_BACKENDS) && @warn "No functional GPU (CUDA or AMD) detected; only \
    RES_BENCH_BACKENDS=CPU will produce results."

# Provenance stamped onto every row so appended multi-node HPC runs stay separable.
device_label(bname) = get(DEVICE_LABELS, bname, "unknown")
const HOSTNAME = gethostname()

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
    push!(available, "CUDA" => CUDA.CUDABackend(), "AMD" => AMDGPU.ROCBackend(), "CPU" => KA.CPU())
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

# --- Append-mode CSV writer ------------------------------------------------------
# APPEND rather than overwrite so results gathered across many HPC compute nodes —
# each job running on its own GPU, writing to a shared file on a common filesystem —
# accumulate into ONE table instead of each run clobbering the last. Every row already
# carries host/device provenance columns (see run_sweep), so the appended rows stay
# attributable and de-dupable downstream.
#
# The header is written only when the file is first created. On a pre-existing file we
# verify the stored header matches the one we are about to append under; a mismatch
# means the file was written by an older/different schema, which we refuse to corrupt
# by appending misaligned columns — rename or delete it (or point RES_BENCH_CSV
# elsewhere). Concurrent jobs on a shared filesystem should therefore either target
# distinct files (per-node RES_BENCH_CSV) or tolerate interleaved appends; a single
# writedlm append call is atomic enough for line-oriented rows in practice.
function append_rows!(out, header, rows)
    exists = isfile(out) && filesize(out) > 0
    if exists
        existing = open(readline, out)
        wanted   = chomp(sprint(io -> writedlm(io, header, ',')))
        existing == wanted || error("""
            CSV schema mismatch: $out already exists with a different header.
              existing: $existing
              expected: $wanted
            Appending would misalign columns. Remove/rename the old file (or set a
            different RES_BENCH_CSV) and re-run.""")
    end
    open(out, "a") do io
        exists || writedlm(io, header, ',')      # header only on first creation
        writedlm(io, permutedims(hcat(rows...)), ',')
    end
    return out
end

# --- Generic sweep driver --------------------------------------------------------
# Sweeps the selected backends × resolutions for one `mode`, timing each case with
# `benchmark_case`, and APPENDS the rows to `csv_name` (see append_rows!). Each row is
# stamped with the host node, the concrete device string, and a UTC timestamp so runs
# collected from many HPC compute nodes into one shared file stay attributable.
# `per_backend_hook(backend, resolutions)` runs once per backend before its cases (the
# AD script uses it to raise the device malloc heap); it defaults to a no-op.
#
# The output path defaults to `csv_name` next to this script but can be redirected with
# RES_BENCH_CSV — e.g. to a shared scratch path that every node's job appends to, or to
# a per-node file to avoid concurrent writers.
function run_sweep(; mode, nsteps, setup_fn, reset!, run!, csv_name,
                     per_backend_hook = (backend, resolutions) -> nothing)
    backends    = selected_backends()
    resolutions = selected_resolutions()
    stamp       = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")

    rows = Vector{Any}[]
    for (bname, backend) in backends
        per_backend_hook(backend, resolutions)
        device = device_label(bname)
        println("\n=== [$bname] mode=$mode  integrator=$(INTEGRATOR)  nsteps=$nsteps  samples=$SAMPLES  host=$HOSTNAME  device=$device ===")

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
            push!(rows, Any[HOSTNAME, device, stamp, PROBLEM, res.name, bname, mode,
                            ncells, nsteps, s.n,
                            s.min, s.min / nsteps, s.median, s.mean, s.std, loss])
            GC.gc()
        end
    end

    isempty(rows) && (println("\nNo successful cases — nothing written."); return nothing)

    # runtime_s / s_per_step are the MINIMUM over samples (the standard BenchmarkTools
    # estimator: least noise-contaminated); median/mean/std are also recorded. host/
    # device/timestamp identify which node+GPU produced each row (appended-file safe).
    header = ["host" "device" "timestamp" "problem" "res" "backend" "mode" "ncells" "nsteps" "samples" "runtime_s" "s_per_step" "median_s" "mean_s" "std_s" "loss"]
    out    = get(ENV, "RES_BENCH_CSV", joinpath(@__DIR__, csv_name))
    append_rows!(out, header, rows)
    println("\nAppended $(length(rows)) row(s) to $out")
    return out
end
