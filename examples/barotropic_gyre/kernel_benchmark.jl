# Per-kernel resolution-scaling benchmark (barotropic gyre)
# =========================================================
# The resolution_scaling_benchmark.jl harness times the WHOLE forward/AD loop, so a
# slow kernel is invisible inside the aggregate. This harness instead times each
# individual compute kernel that a single forward step launches, at every available
# mesh resolution, so suboptimal GPU kernels (launch-overhead-bound, low-occupancy,
# or badly-scaling) stand out on their own.
#
# One forward barotropic-gyre step decomposes into these kernels (grouped by stage):
#
#   diagnostics        (diagnostic_compute!)
#     thicknessFlux           nEdges     u * h_edge
#     velocityDivCell         nCells     divergence of thicknessFlux  (Div op, 2 kernels)
#     relativeVorticity       nVertices  curl of normalVelocity
#     layerThicknessEdge      nEdges     cell->edge interpolation
#   normalVelocity tendency  (computeNormalVelocityTendency!)
#     zeroTendNormalVel       nEdges
#     pressureGradient        nEdges     -g d(ssh)/dx        (Gradient op)
#     advectionCoriolis       nEdges     linear Coriolis
#     momentumMixing          nEdges     del2 viscosity
#     windForcing             nEdges     surface wind stress
#   layerThickness tendency  (computeLayerThicknessTendency!)
#     zeroTendLayerThk        nCells
#     thicknessFluxDiv        nCells     divergence of thicknessFlux
#   time integration
#     advanceTimeLevels       fields     copy [end] -> [end-1]
#
# For context the harness ALSO times the three composite drivers (diagnostic_compute!,
# computeNormalVelocityTendency!, computeLayerThicknessTendency!) and one full
# ocn_timestep, so per-kernel numbers can be checked against the stage totals.
#
# GPU timing: a kernel launch is asynchronous, so the timed body launches the kernel
# and THEN KA.synchronize(backend)s — otherwise we would measure host launch latency,
# not device execution. Each sample is preceded (untimed) by an identical state reset
# (restore the initial prognostic fields + recompute diagnostics) so every sample times
# the same work, and mutating/accumulating kernels never drift toward Inf across samples.
# The CSV reports the BenchmarkTools MINIMUM (least noise-contaminated) plus median/mean.
#
# Run with:  julia --project=. examples/barotropic_gyre/kernel_benchmark.jl
#
# Environment overrides (all optional):
#   KBENCH_BACKENDS=GPU        # or =CPU, or =GPU,CPU        (default: both)
#   KBENCH_RES=5km,10km,20km,40km       # subset/order of resolutions  (default: all found)
#   KBENCH_SAMPLES=200         # timed samples per kernel      (default 200)
#   KBENCH_SECONDS=30          # BenchmarkTools budget/kernel  (default 30)
#   KBENCH_REPEATS=1           # kernel launches per sample (amortizes launch overhead)
#   KBENCH_DEVICE=1            # CUDA device index             (default 1)

using Dates
import KernelAbstractions as KA
using BenchmarkTools
using MOKA
using CUDA
import CUDA: @allowscalar
using DelimitedFiles
using Printf

CUDA.device!(parse(Int, get(ENV, "KBENCH_DEVICE", "1")))

# --- Resolution ladder (barotropic gyre) --------------------------------------------
# Only directories that actually contain a config are swept; ncells/nEdges are read
# from the mesh at run time and written to the CSV.
const BG = @__DIR__
const ALL_RES = [
    (name = "40km", dir = joinpath(BG, "40km"), config = "config.yml"),
    (name = "20km", dir = joinpath(BG, "20km"), config = "config.yml"),
    (name = "10km", dir = joinpath(BG, "10km"), config = "config.yml"),
    (name = "5km",  dir = joinpath(BG, "5km"),  config = "config.yml"),
    (name = "2.5km", dir = joinpath(BG, "2.5km"), config = "config.yml"),
]

function selected_resolutions()
    found = filter(r -> isfile(joinpath(r.dir, r.config)), ALL_RES)
    haskey(ENV, "KBENCH_RES") || return found
    want = split(ENV["KBENCH_RES"], ',')
    return [r for w in want for r in found if r.name == w]
end

function selected_backends()
    all = ["GPU" => CUDABackend(), "CPU" => KA.CPU()]
    haskey(ENV, "KBENCH_BACKENDS") || return all
    want = split(ENV["KBENCH_BACKENDS"], ',')
    return filter(p -> first(p) in want, all)
end

const SAMPLES = parse(Int,     get(ENV, "KBENCH_SAMPLES", "200"))
const SECONDS = parse(Float64, get(ENV, "KBENCH_SECONDS", "30"))
const REPEATS = parse(Int,     get(ENV, "KBENCH_REPEATS", "1"))

_dt_seconds(Setup) = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

# --- Kernel catalogue ---------------------------------------------------------------
# Each entry: (group, name, ndrange-field, apply). `ndrange` labels the parallel
# extent the kernel loops over (nEdges/nCells/nVertices) so per-element throughput can
# be derived. `apply(st)` launches the kernel(s) but does NOT synchronize — the timed
# wrapper does that once, after REPEATS launches.
const nV = MOKA.normalVelocity
const lT = MOKA.layerThickness

struct Kernel
    group::String
    name::String
    extent::Symbol            # :nEdges | :nCells | :nVertices | :mixed
    apply::Function
end

const KERNELS = Kernel[
    # diagnostics
    Kernel("diagnostics", "thicknessFlux",      :nEdges,    st -> MOKA.calculate_thicknessFlux!(st.Diag, st.Prog, st.Mesh)),
    Kernel("diagnostics", "velocityDivCell",    :nCells,    st -> MOKA.calculate_velocityDivCell!(st.Diag, st.Prog, st.Mesh)),
    Kernel("diagnostics", "relativeVorticity",  :nVertices, st -> MOKA.calculate_relativeVorticity!(st.Diag, st.Prog, st.Mesh)),
    Kernel("diagnostics", "layerThicknessEdge", :nEdges,    st -> MOKA.calculate_layerThicknessEdge!(st.Diag, st.Prog, st.Mesh)),
    # normalVelocity tendency terms
    Kernel("normalVelTend", "pressureGradient",   :nEdges, st -> nV.pressure_gradient_tendency!(st.Tend, st.Prog, st.Diag, st.Mesh, nV.sshGradient)),
    Kernel("normalVelTend", "advectionCoriolis",  :nEdges, st -> nV.horizontal_advection_and_coriolis_tendency!(st.Tend, st.Prog, st.Diag, st.Mesh, nV.linearCoriolis)),
    Kernel("normalVelTend", "momentumMixing",     :nEdges, st -> nV.horizontal_momentum_mixing_tendency!(st.Tend, st.Prog, st.Diag, st.Mesh, nV.Del2)),
    Kernel("normalVelTend", "windForcing",        :nEdges, st -> nV.wind_forcing_tendency!(st.Tend, st.Diag, st.Mesh)),
    # layerThickness tendency
    Kernel("layerThkTend", "thicknessFluxDiv",    :nCells, st -> lT.horizontal_advection_tendency!(st.Tend, st.Prog, st.Diag, st.Mesh)),
    # time integration
    Kernel("integration", "advanceTimeLevels",    :mixed,  st -> MOKA.advance_time_levels!(st.Prog)),
    # composite drivers (for cross-checking against the sum of their parts)
    Kernel("composite", "diagnostic_compute!",           :mixed, st -> MOKA.diagnostic_compute!(st.Mesh, st.Diag, st.Prog)),
    Kernel("composite", "computeNormalVelocityTendency!", :nEdges, st -> MOKA.computeNormalVelocityTendency!(st.Tend, st.Prog, st.Diag, st.Mesh)),
    Kernel("composite", "computeLayerThicknessTendency!", :nCells, st -> MOKA.computeLayerThicknessTendency!(st.Tend, st.Prog, st.Diag, st.Mesh)),
    Kernel("composite", "ocn_timestep",                   :mixed, st -> ocn_timestep(st.timestep, st.Prog, st.Diag, st.Tend, st.Mesh, st.integrator)),
]

# --- State setup / reset ------------------------------------------------------------
snapshot_prog(Prog) = (ssh            = map(copy, Prog.ssh),
                       normalVelocity = map(copy, Prog.normalVelocity),
                       layerThickness = map(copy, Prog.layerThickness))

function restore_prog!(Prog, snap)
    foreach(copyto!, Prog.ssh,            snap.ssh)
    foreach(copyto!, Prog.normalVelocity, snap.normalVelocity)
    foreach(copyto!, Prog.layerThickness, snap.layerThickness)
    return nothing
end

function setup_state(dir, config, backend)
    cd(dir)
    Setup, Diag, Tend, Prog = ocn_init(config; backend=backend)
    Mesh = Setup.mesh
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = _dt_seconds(Setup)
    integrator = parse_integrator(
        MOKA.ConfigGet(MOKA.ConfigGet(Setup.config.namelist, "time_integration"),
                       "config_time_integrator"))
    edges    = Mesh.HorzMesh.Edges
    cells    = Mesh.HorzMesh.PrimaryCells
    vertices = Mesh.HorzMesh.DualCells
    return (; Prog, Diag, Tend, Mesh, timestep, integrator, backend,
              snap = snapshot_prog(Prog),
              nEdges = edges.nEdges, nCells = cells.nCells,
              nVertices = getfield(vertices, :nVertices))
end

# Untimed per-sample reset: restore the initial prognostic fields and repopulate the
# diagnostics so every tendency kernel sees identical, valid inputs each sample.
function reset_state!(st)
    restore_prog!(st.Prog, st.snap)
    MOKA.diagnostic_compute!(st.Mesh, st.Diag, st.Prog)
    KA.synchronize(st.backend)
    return nothing
end

extent_count(st, e) = e === :nEdges    ? st.nEdges :
                      e === :nCells    ? st.nCells :
                      e === :nVertices ? st.nVertices : 0

# --- Timed body ---------------------------------------------------------------------
# Launch the kernel REPEATS times, then synchronize ONCE. Reported time is divided by
# REPEATS downstream so it is per-launch. On GPU this amortizes host launch latency
# into the device-execution estimate; on CPU it is simply REPEATS iterations.
function timed!(apply, st)
    @inbounds for _ in 1:REPEATS
        apply(st)
    end
    KA.synchronize(st.backend)
    return nothing
end

function benchmark_kernel(k::Kernel, st)
    apply = k.apply
    timed!(apply, st)                    # warm-up: JIT compile, untimed
    bench = @benchmarkable timed!($apply, $st) setup = (reset_state!($st)) evals = 1 samples = SAMPLES seconds = SECONDS
    return run(bench)
end

ns_per_launch(trial) = (min    = minimum(trial).time / REPEATS,
                        median = BenchmarkTools.median(trial).time / REPEATS,
                        mean   = BenchmarkTools.mean(trial).time / REPEATS,
                        n      = length(trial.times))

function main()
    backends    = selected_backends()
    resolutions = selected_resolutions()
    isempty(resolutions) && error("No barotropic-gyre resolutions found to benchmark.")

    rows = Vector{Any}[]
    for (bname, backend) in backends
        for res in resolutions
            print("\n=== [$bname] $(res.name): init + warm-up... "); flush(stdout)
            st = setup_state(res.dir, res.config, backend)
            @printf("nEdges=%d nCells=%d nVertices=%d ===\n", st.nEdges, st.nCells, st.nVertices)
            @printf("  %-13s %-30s %10s  %12s  %12s  %10s\n",
                    "group", "kernel", "extent", "min(µs)", "median(µs)", "ns/elem")
            for k in KERNELS
                trial = try
                    benchmark_kernel(k, st)
                catch err
                    @printf("  %-13s %-30s  FAILED: %s\n", k.group, k.name, sprint(showerror, err))
                    continue
                end
                s   = ns_per_launch(trial)
                ext = extent_count(st, k.extent)
                nspe = ext > 0 ? s.min / ext : NaN
                @printf("  %-13s %-30s %10d  %12.3f  %12.3f  %10.4f\n",
                        k.group, k.name, ext, s.min/1e3, s.median/1e3, nspe)
                push!(rows, Any[bname, res.name, st.nEdges, st.nCells, st.nVertices,
                                k.group, k.name, string(k.extent), ext, s.n,
                                s.min, s.median, s.mean, nspe])
                GC.gc()
            end
        end
    end

    header = ["backend" "res" "nEdges" "nCells" "nVertices" "group" "kernel" "extent" "extent_count" "samples" "min_ns" "median_ns" "mean_ns" "ns_per_elem"]
    data   = permutedims(hcat(rows...))
    out    = joinpath(BG, "kernel_benchmark.csv")
    open(out, "w") do io
        writedlm(io, header, ',')
        writedlm(io, data, ',')
    end
    println("\nWrote $out")
end

main()
