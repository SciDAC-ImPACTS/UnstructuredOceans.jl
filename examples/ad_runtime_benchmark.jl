# AD runtime benchmark
# =====================
# Times the checkpointed reverse-mode AD driver (`ocn_run_loop_checkpointed!`)
# for two example problems — the barotropic gyre and the inertial-gravity wave —
# on both the GPU and the CPU, across a range of stop times (number of timesteps),
# and writes the results to a CSV that `plot_ad_runtime_benchmark.jl` turns into
# figures.
#
# "Stop time" is expressed as a number of timesteps: for each case we override the
# simulation-end alarm to ring exactly `nsteps` steps after the start, so the two
# cases are compared on equal step counts even though their dt differs (gyre dt=1s,
# IGW dt=300s).
#
# Backends swept: GPU (CUDABackend) and CPU (KernelAbstractions.CPU). Override with
# AD_BENCH_BACKENDS=GPU or =CPU to run just one (default: both). CPU AD is much
# slower than GPU, so you may also want a shorter AD_BENCH_NSTEPS for CPU-heavy runs.
#
# Run with:  julia --project=. examples/ad_runtime_benchmark.jl

using Dates
import KernelAbstractions as KA
using Enzyme
using MOKA
using CUDA
import CUDA: @allowscalar
using DelimitedFiles
using Printf

# --- Cases to benchmark (mirrors the per-example enzyme_test.jl harnesses) ------
const CASES = (
    (name = "barotropic_gyre",
     dir  = joinpath(@__DIR__, "barotropic_gyre", "10km"),
     config = "enzyme_config.yml"),
    (name = "inertial_gravity_wave",
     dir  = joinpath(@__DIR__, "inertial_gravity_wave", "200km"),
     config = "config_enzyme.yml"),
)

# Stop times to sweep, as step counts. Kept modest so the whole sweep is minutes,
# not hours; extend the upper end if you want to probe scaling further. Override
# from the shell for a quick smoke test, e.g. AD_BENCH_NSTEPS=1,2 julia ...
const NSTEPS = haskey(ENV, "AD_BENCH_NSTEPS") ?
    parse.(Int, split(ENV["AD_BENCH_NSTEPS"], ',')) : [1, 2, 4, 8, 16, 32, 64]
const CELL   = 5   # cell/edge index whose gradient we record (matches enzyme_test.jl)

# Backends to sweep: name => KA backend. CUDABackend for GPU, KA.CPU() for CPU.
function selected_backends()
    all = ["GPU" => CUDABackend(), "CPU" => KA.CPU()]
    haskey(ENV, "AD_BENCH_BACKENDS") || return all
    want = split(ENV["AD_BENCH_BACKENDS"], ',')
    return filter(p -> first(p) in want, all)
end

# Override the simulation-end alarm so the run stops after exactly `nsteps` steps.
# advance! increments currTime by one timeStep per iteration and the alarm rings on
# exact equality, so startTime + nsteps*timeStep lands precisely on step `nsteps`.
function set_nsteps!(clock, simulationAlarm, nsteps)
    simulationAlarm.ringTime = clock.startTime + nsteps * clock.timeStep
    simulationAlarm.ringing  = false
    simulationAlarm.stopped  = false
    return nothing
end

# Build a fresh model state, then time ONLY the AD driver call (init / mesh I/O is
# excluded). Returns (runtime_s, loss, d_layer, d_vel, dt_s).
function time_ad(dir, config, nsteps, backend)
    cd(dir)
    Setup, Diag, Tend, Prog = ocn_init(config; backend=backend)
    clock, simAlarm, outAlarm = ocn_init_alarms(Setup)
    set_nsteps!(clock, simAlarm, nsteps)

    timestep   = KA.zeros(backend, Float64, (1,))
    d_timestep = KA.zeros(backend, Float64, (1,))
    sumGPU     = KA.zeros(backend, Float64, (1,))
    d_sumGPU   = KA.ones(backend,  Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    Mesh   = Setup.mesh
    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)

    KA.synchronize(backend)
    runtime = @elapsed begin
        ocn_run_loop_checkpointed!(sumGPU, d_sumGPU, timestep, d_timestep,
                                   Prog, d_Prog, Diag, d_Diag, Tend, d_Tend,
                                   Mesh, ForwardEuler,
                                   clock, simAlarm, outAlarm)
        KA.synchronize(backend)
    end

    loss   = @allowscalar sumGPU[1]
    dlayer = @allowscalar d_Prog.layerThickness[end][1, CELL]
    dvel   = @allowscalar d_Prog.normalVelocity[end][1, CELL]
    dt_s   = @allowscalar timestep[1]
    return runtime, loss, dlayer, dvel, dt_s
end

function main()
    backends = selected_backends()

    rows = Vector{Any}[]
    for (bname, backend) in backends
        # Checkpointed AD still needs the raised device malloc heap for a single
        # step's tape (see set_ad_device_heap! docstring); it is now horizon-
        # independent. No-op on CPU.
        set_ad_device_heap!(backend)

        for case in CASES
            println("\n=== [$bname] Case: $(case.name)  ($(case.config)) ===")
            # Warm-up at nsteps=1 to pay the Julia/Enzyme/CUDA compilation cost
            # ONCE, outside the timed measurements.
            print("  warming up (compilation)... "); flush(stdout)
            time_ad(case.dir, case.config, 1, backend)
            GC.gc()
            println("done")

            for n in NSTEPS
                rt, loss, dl, dv, dt_s = time_ad(case.dir, case.config, n, backend)
                GC.gc()
                @printf("  nsteps=%4d  dt=%6.1fs  runtime=%8.3fs  (%.3f s/step)  loss=%.6e\n",
                        n, dt_s, rt, rt / n, loss)
                push!(rows, Any[case.name, bname, n, dt_s, rt, rt / n, loss, dl, dv])
            end
        end
    end

    # --- Write CSV --------------------------------------------------------------
    header = ["case" "backend" "nsteps" "dt_s" "runtime_s" "s_per_step" "loss" "d_layer" "d_vel"]
    data   = permutedims(hcat(rows...))
    out    = joinpath(@__DIR__, "ad_runtime_benchmark.csv")
    open(out, "w") do io
        writedlm(io, header, ',')
        writedlm(io, data, ',')
    end
    println("\nWrote $out")
end

main()
