# Reverse-mode AD / primal runtime ratio
# =======================================
# Joins the forward CSV (resolution_scaling_benchmark.csv) and the checkpointed
# reverse-mode AD CSV (resolution_scaling_ad_benchmark.csv) — both written by the
# resolution-scaling benchmark scripts with an identical schema — and reports the
# AD *overhead factor*: how many times more expensive one differentiated step is
# than one primal step.
#
#     ratio = (AD s/step) / (forward s/step)
#
# The PER-STEP runtime is used (not total) so the two sweeps are comparable even
# though they run different step counts (forward nsteps=8, AD nsteps=4 by default):
# dividing per-step costs cancels the horizon. Rows are matched on
# (problem, backend, device, integrator, ncells) so a ratio only ever compares the
# forward and adjoint of the SAME problem/resolution on the SAME hardware.
#
# Both CSVs are append-mode and may hold repeats (rerun / multiple HPC nodes) and,
# for AD, a horizon sweep (several nsteps). Each side is collapsed to the MIN s/step
# per match key (the standard low-noise BenchmarkTools estimator); for the AD side a
# single horizon is pinned first (largest nsteps = best steady-state per-step estimate,
# override with RES_PLOT_NSTEPS) so the ratio is not contaminated by mixing checkpointing
# regimes.
#
# Outputs:
#   - a printed table (one line per matched case)
#   - resolution_scaling_ad_primal_ratio.csv  (the joined ratios, for the paper)
#   - resolution_scaling_ad_primal_ratio.png  (ratio vs cell count, one line per backend)
#
# Run with:  julia examples/plot_ad_primal_ratio.jl
# Needs only CairoMakie + DelimitedFiles (not MOKA) — run it in an environment that has
# CairoMakie (e.g. the global env; add with Pkg.add("CairoMakie")).
#
# Environment overrides (all optional):
#   RES_PLOT_DEVICE=<string>   # pin one concrete device string (see note below)
#   RES_PLOT_NSTEPS=<int>      # AD horizon to use (default: largest present)

using CairoMakie
using DelimitedFiles
using Printf

const FWD_CSV   = joinpath(@__DIR__, "resolution_scaling_benchmark.csv")
const AD_CSV    = joinpath(@__DIR__, "resolution_scaling_ad_benchmark.csv")
const RATIO_CSV = joinpath(@__DIR__, "resolution_scaling_ad_primal_ratio.csv")
const RATIO_PNG = joinpath(@__DIR__, "resolution_scaling_ad_primal_ratio.png")

# Backend LABELs ("CUDA") can span physically different cards across HPC nodes. Set
# RES_PLOT_DEVICE to pin one; otherwise a device mismatch across the two CSVs at the
# same match key is WARNed about (the ratio would then divide costs from two machines).
const PLOT_DEVICE = get(ENV, "RES_PLOT_DEVICE", "")

# --- CSV loading ----------------------------------------------------------------
# Pull the columns we need out of one CSV; returns empty vectors if it is absent.
# Tolerant of older schemas (problem/device/nsteps/integrator all have fallbacks).
function load_csv(path)
    isfile(path) || return (backend = String[], device = String[], problem = String[],
                            res = String[], integrator = String[], ncells = Float64[],
                            sstep = Float64[], nsteps = Float64[])
    raw, header = readdlm(path, ',', header = true)
    header = vec(header)
    n = size(raw, 1)
    col(name) = raw[:, findfirst(==(name), header)]
    has(name) = name in header
    problem    = has("problem")    ? string.(col("problem"))    : fill("igw", n)
    device     = has("device")     ? string.(col("device"))     : fill("unknown", n)
    res        = has("res")        ? string.(col("res"))        : fill("?", n)
    integrator = has("integrator") ? string.(col("integrator")) : fill("?", n)
    nsteps     = has("nsteps")     ? Float64.(col("nsteps"))    : fill(1.0, n)
    return (backend = string.(col("backend")), device = device, problem = problem,
            res = res, integrator = integrator, ncells = Float64.(col("ncells")),
            sstep = Float64.(col("s_per_step")), nsteps = nsteps)
end

# Optionally restrict to a single device string.
function device_filter(d)
    isempty(PLOT_DEVICE) && return trues(length(d.backend))
    return d.device .== PLOT_DEVICE
end

# Pin one AD horizon so the per-step cost reflects a single checkpointing regime.
# Forward per-step is horizon-independent, so it is left untouched.
function pin_horizon(mask, d)
    horizons = sort(unique(d.nsteps[mask]))
    length(horizons) <= 1 && return mask
    want = haskey(ENV, "RES_PLOT_NSTEPS") ? parse(Float64, ENV["RES_PLOT_NSTEPS"]) :
           maximum(horizons)
    return mask .& (d.nsteps .== want)
end

# Collapse to MIN s/step per match key, tracking every device seen behind each key so a
# min taken across mixed hardware can be flagged downstream. Returns
# key => (sstep, res, integrator, devices), keyed on (problem, backend, ncells).
function collapse(d, mask)
    best = Dict{Tuple{String,String,Float64},NamedTuple}()
    for i in eachindex(d.backend)
        mask[i] || continue
        key = (d.problem[i], d.backend[i], d.ncells[i])
        cur = get(best, key, nothing)
        devices = cur === nothing ? Set{String}() : cur.devices
        push!(devices, d.device[i])
        if cur === nothing || d.sstep[i] < cur.sstep
            best[key] = (sstep = d.sstep[i], res = d.res[i],
                         integrator = d.integrator[i], devices = devices)
        else
            best[key] = (; cur..., devices = devices)  # keep min, updated device set
        end
    end
    return best
end

# --- Load, filter, collapse -----------------------------------------------------
fwd = load_csv(FWD_CSV)
ad  = load_csv(AD_CSV)
isempty(fwd.backend) && error("$FWD_CSV not found — run the forward benchmark first.")
isempty(ad.backend)  && error("$AD_CSV not found — run the AD benchmark first.")

fwd_best = collapse(fwd, device_filter(fwd))
ad_best  = collapse(ad,  pin_horizon(device_filter(ad), ad))

# --- Join on (problem, backend, ncells) -----------------------------------------
keys_common = sort(collect(intersect(keys(fwd_best), keys(ad_best)));
                   by = k -> (k[1], k[2], k[3]))
isempty(keys_common) && error("""
    No overlapping (problem, backend, ncells) rows between the forward and AD CSVs —
    nothing to ratio. The forward CSV must contain the same problem/backend/resolution
    as the AD CSV. (Forward currently: $(unique(fwd.backend)); AD: $(unique(ad.backend)).)""")

probtitle = Dict("gyre" => "barotropic gyre", "igw" => "inertial-gravity wave")
probname_of(p) = get(probtitle, p, p)

# --- Table + ratio CSV ----------------------------------------------------------
rows = Vector{Any}[]
println("\nReverse-mode AD / primal per-step runtime ratio")
println("(ratio = AD s/step ÷ forward s/step; larger ⇒ costlier adjoint)\n")
@printf("%-6s %-7s %-7s %-12s %9s %12s %12s %8s\n",
        "prob", "backend", "res", "integrator", "ncells", "fwd s/step", "ad s/step", "ratio")
println(repeat("-", 82))
for k in keys_common
    problem, backend, ncells = k
    f = fwd_best[k]; a = ad_best[k]
    ratio = a.sstep / f.sstep
    # Warn if either side min-collapsed across distinct hardware (or the two sides used
    # different devices) — the ratio would then mix machines.
    alldevs = union(f.devices, a.devices)
    length(alldevs) > 1 && @warn "match $(problem)/$(backend)/N=$(Int(ncells)) spans \
        multiple devices $(collect(alldevs)) — ratio mixes hardware; set RES_PLOT_DEVICE."
    @printf("%-6s %-7s %-7s %-12s %9d %12.6f %12.6f %8.2f×\n",
            problem, backend, a.res, a.integrator, Int(ncells), f.sstep, a.sstep, ratio)
    push!(rows, Any[problem, backend, join(sort(collect(alldevs)), "|"), a.res,
                    a.integrator, Int(ncells), f.sstep, a.sstep, ratio])
end

header = ["problem" "backend" "device" "res" "integrator" "ncells" "fwd_s_per_step" "ad_s_per_step" "ratio"]
open(RATIO_CSV, "w") do io
    writedlm(io, header, ',')
    writedlm(io, permutedims(hcat(rows...)), ',')
end
println("\nWrote $RATIO_CSV")

# --- Figure: ratio vs cell count, one line per backend --------------------------
bmarker = Dict("CUDA" => :circle, "AMD" => :diamond, "GPU" => :circle, "CPU" => :utriangle)
bcolor  = Dict("CUDA" => :dodgerblue3, "AMD" => :firebrick3, "GPU" => :dodgerblue3,
               "CPU" => :darkorange2)

problem  = first(keys_common)[1]
probname = probname_of(problem)
backends = unique(k[2] for k in keys_common)

fig = Figure(size = (640, 480))
ax = Axis(fig[1, 1];
          title  = "Reverse-mode AD overhead ($probname)",
          xlabel = "number of cells  N",
          ylabel = "AD s/step ÷ forward s/step  [×]",
          xscale = log10, yscale = log10)

for b in backends
    ks = sort([k for k in keys_common if k[2] == b]; by = k -> k[3])
    xs = Float64[k[3] for k in ks]
    ys = Float64[ad_best[k].sstep / fwd_best[k].sstep for k in ks]
    isempty(xs) && continue
    scatterlines!(ax, xs, ys;
                  color = get(bcolor, b, :black),
                  marker = get(bmarker, b, :diamond), markersize = 11,
                  linewidth = 2, label = b)
end

hlines!(ax, [1.0]; color = (:gray, 0.6), linestyle = :dash, label = "1× (primal cost)")
axislegend(ax; position = :lt, framevisible = true, labelsize = 11)

save(RATIO_PNG, fig)
println("Wrote $RATIO_PNG")
