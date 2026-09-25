# Resolution-scaling benchmark for checkpointed reverse-mode AD.
#
# Times the barotropic gyre or inertial-gravity-wave adjoint on selected GPUs.
# Initialization and compilation are outside the timed region. Each BenchmarkTools
# sample restores the initial state and clears Enzyme's shadows before running.
#
# Run: julia --project=. examples/resolution_scaling_ad_benchmark.jl
#
# Key overrides:
#   RES_BENCH_PROBLEM=gyre|igw        RES_BENCH_BACKENDS=CUDA,AMD,CPU
#   RES_BENCH_RES=40km,20km           RES_BENCH_AD_NSTEPS=1,2,4,8
#   RES_BENCH_AD_NSNAPS=4             RES_BENCH_INTEGRATOR=RK4
#   RES_BENCH_SAMPLES=5               RES_BENCH_SECONDS=600
#   RES_BENCH_CSV=/path/to/results.csv
#
# resolution_scaling_common.jl provides backend discovery, selection, sampling,
# and append-only CSV output.

using Dates
import KernelAbstractions as KA
using BenchmarkTools
using Enzyme
using Checkpointing
using UnstructuredOceans
using GPUArraysCore: @allowscalar
using DelimitedFiles
using Printf

include(joinpath(@__DIR__, "resolution_scaling_common.jl"))

const AD_NSTEPS = parse.(Int, split(get(ENV, "RES_BENCH_AD_NSTEPS", "8"), ','))
const AD_NSNAPS = parse(Int, get(ENV, "RES_BENCH_AD_NSNAPS", "4"))

# Enzyme allocates reverse tapes from the device-side malloc heap. RK4 records
# four tendency evaluations per step, so it needs four times the Euler floor.
const HEAP_FLOOR_BYTES = 512 * 1024 * 1024
const HEAP_REFERENCE_CELLS = 40_000
const HEAP_SAFETY_FACTOR = 2

stage_count(::Type{ForwardEuler}) = 1
stage_count(::Type{RungeKutta4}) = 4

function ad_heap_bytes(ncells, integrator)
    floor = HEAP_FLOOR_BYTES * stage_count(integrator)
    scaled = HEAP_SAFETY_FACTOR * floor * ncells / HEAP_REFERENCE_CELLS
    return max(floor, round(Int, scaled))
end

function setup_ad(dir, config, nsteps, backend)
    return cd(dir) do
        Setup, Diag, Tend, Prog = ocn_init(config; backend)
        integrator = case_integrator(Setup)
        timestep = KA.zeros(backend, Float64, 1)
        @allowscalar timestep[1] = _dt_seconds(Setup)

        model = OceanModel(integrator, Prog, Diag, Tend, Setup.mesh, timestep)
        d_model = OceanModel(integrator,
                             Enzyme.make_zero(Prog),
                             Enzyme.make_zero(Diag),
                             Enzyme.make_zero(Tend),
                             Setup.mesh,
                             Enzyme.make_zero(timestep))

        return (; model, d_model, backend, nsteps,
                nsnaps = clamp(AD_NSNAPS, 1, nsteps),
                integrator,
                dt_s = _dt_seconds(Setup),
                ncells = length(Prog.ssh[end]),
                snapshot = snapshot_prog(Prog))
    end
end

function reset_ad!(state)
    restore_prog!(state.model.Prog, state.snapshot)
    Enzyme.make_zero!(state.d_model.Prog)
    Enzyme.make_zero!(state.d_model.Diag)
    Enzyme.make_zero!(state.d_model.Tend)
    Enzyme.make_zero!(state.d_model.dt)
    KA.synchronize(state.backend)
    return nothing
end

function run_ad!(state)
    loss = ocn_run_loop_checkpointed!(state.model, state.d_model, state.nsteps;
                                      nsnaps = state.nsnaps, verbose = 0)
    KA.synchronize(state.backend)
    return loss
end

function configure_ad_heap!(backend, resolutions)
    bytes = maximum(resolution -> ad_heap_bytes(
        resolution.ncells,
        config_integrator(resolution.dir, resolution.config),
    ), resolutions)
    set_ad_device_heap!(backend; bytes)
    return nothing
end

run_sweep(; mode = "ad",
            nsteps = AD_NSTEPS,
            setup_fn = setup_ad,
            reset! = reset_ad!,
            run! = run_ad!,
            csv_name = "resolution_scaling_ad_benchmark.csv",
            per_backend_hook = configure_ad_heap!
        )
