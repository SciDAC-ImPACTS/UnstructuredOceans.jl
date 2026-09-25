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
# (problem, backend, res, integrator, ncells), so a ratio only compares equivalent
# forward and adjoint cases. Device names are retained to warn about mixed hardware.
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
# Needs only CairoMakie + DelimitedFiles (not UnstructuredOceans) — run it in an environment that has
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
const PLOT_PROBLEM = get(ENV, "RES_PLOT_PROBLEM", "")

# --- CSV loading ----------------------------------------------------------------
# Pull the columns needed for the ratio from one CSV. Older schemas may omit
# problem, device, res, integrator, and nsteps.
function load_csv(path)
    isfile(path) || return (backend = String[], device = String[], problem = String[],
                            res = String[], integrator = String[], ncells = Float64[],
                            sstep = Float64[], nsteps = Float64[])
    raw, header = readdlm(path, ',', header = true)
    header = vec(header)
    n = size(raw, 1)
    function col(name)
        index = findfirst(==(name), header)
        isnothing(index) && error("$path is missing required column '$name'.")
        return raw[:, index]
    end
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

# Optionally restrict to a single device string and/or a single problem. The CSVs
# are append-only and may hold several problems; RES_PLOT_PROBLEM selects one so the
# single-problem plot below has an unambiguous dataset.
function device_filter(d)
    mask = trues(length(d.backend))
    isempty(PLOT_DEVICE)  || (mask .&= d.device .== PLOT_DEVICE)
    isempty(PLOT_PROBLEM) || (mask .&= d.problem .== PLOT_PROBLEM)
    return mask
end

# Pin one AD horizon so the per-step cost reflects a single checkpointing regime.
# Forward per-step is horizon-independent, so it is left untouched.
function pin_horizon(mask, d)
    horizons = sort(unique(d.nsteps[mask]))
    length(horizons) <= 1 && return mask
    want = haskey(ENV, "RES_PLOT_NSTEPS") ? parse(Float64, ENV["RES_PLOT_NSTEPS"]) :
           maximum(horizons)
    want in horizons || error("RES_PLOT_NSTEPS=$want is unavailable; found $(join(horizons, ", ")).")
    return mask .& (d.nsteps .== want)
end

# Collapse to MIN s/step per match key, tracking every device seen behind each key so a
# min taken across mixed hardware can be flagged downstream. Returns
# key => (sstep, devices), keyed on (problem, backend, res, integrator, ncells).
function collapse(d, mask)
    best = Dict{Tuple{String,String,String,String,Float64},NamedTuple}()
    for i in eachindex(d.backend)
        mask[i] || continue
        key = (d.problem[i], d.backend[i], d.res[i], d.integrator[i], d.ncells[i])
        cur = get(best, key, nothing)
        devices = cur === nothing ? Set{String}() : cur.devices
        push!(devices, d.device[i])
        if cur === nothing || d.sstep[i] < cur.sstep
            best[key] = (sstep = d.sstep[i], devices = devices)
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

# --- Join on (problem, backend, res, integrator, ncells) ------------------------
keys_common = sort(collect(intersect(keys(fwd_best), keys(ad_best)));
                   by = k -> (k[1], k[2], k[3], k[4], k[5]))
isempty(keys_common) && error("""
    No overlapping (problem, backend, res, integrator, ncells) rows between the forward
    and AD CSVs. The forward CSV must contain equivalent AD benchmark cases.
    (Forward backends: $(unique(fwd.backend)); AD backends: $(unique(ad.backend)).)""")

problems = unique(first.(keys_common))
length(problems) == 1 || error("This plot supports one problem at a time; found $(join(problems, ", ")).")

probtitle = Dict("gyre" => "barotropic gyre", "igw" => "inertial-gravity wave")
probname_of(p) = get(probtitle, p, p)

# --- Table + ratio CSV ----------------------------------------------------------
rows = NamedTuple[]
println("\nReverse-mode AD / primal per-step runtime ratio")
println("(ratio = AD s/step ÷ forward s/step; larger ⇒ costlier adjoint)\n")
@printf("%-6s %-7s %-7s %-12s %9s %12s %12s %8s\n",
        "prob", "backend", "res", "integrator", "ncells", "fwd s/step", "ad s/step", "ratio")
println(repeat("-", 82))
for k in keys_common
    problem, backend, res, integrator, ncells = k
    f = fwd_best[k]; a = ad_best[k]
    ratio = a.sstep / f.sstep
    # Warn if either side min-collapsed across distinct hardware (or the two sides used
    # different devices) — the ratio would then mix machines.
    alldevs = union(f.devices, a.devices)
    length(alldevs) > 1 && @warn "match $(problem)/$(backend)/N=$(Int(ncells)) spans \
        multiple devices $(collect(alldevs)) — ratio mixes hardware; set RES_PLOT_DEVICE."
    @printf("%-6s %-7s %-7s %-12s %9d %12.6f %12.6f %8.2f×\n",
            problem, backend, res, integrator, Int(ncells), f.sstep, a.sstep, ratio)
    push!(rows, (problem = problem, backend = backend,
                 device = join(sort(collect(alldevs)), "|"), res = res,
                 integrator = integrator, ncells = Int(ncells),
                 fwd_s_per_step = f.sstep, ad_s_per_step = a.sstep, ratio = ratio))
end

open(RATIO_CSV, "w") do io
    writedlm(io, permutedims(collect(string.(propertynames(first(rows))))), ',')
    for row in rows
        writedlm(io, permutedims(collect(values(row))), ',')
    end
end
println("\nWrote $RATIO_CSV")

# --- Figure: ratio vs cell count, one line per backend --------------------------
# Canonical backend styling, shared across all benchmark plots (see plot_kernel_benchmark.jl).
bcolor  = Dict("CPU" => :darkorange2, "CUDA" => :seagreen4, "AMD" => :firebrick3,
               "oneAPI" => :dodgerblue3, "GPU" => :seagreen4)
bmarker = Dict("CPU" => :utriangle, "CUDA" => :circle, "AMD" => :diamond,
               "oneAPI" => :rect, "GPU" => :circle)

problem  = only(problems)
probname = probname_of(problem)
backends = unique(k[2] for k in keys_common)

fig = Figure(size = (640, 360))
ax = Axis(fig[1, 1];
          title  = "Reverse-mode AD overhead ($probname)",
          xlabel = "number of cells  N",
          ylabel = "AD s/step ÷ forward s/step  [×]",
          xscale = log10, yscale = log10)

for b in backends
    ks = sort([k for k in keys_common if k[2] == b]; by = k -> k[5])
    xs = Float64[k[5] for k in ks]
    ys = Float64[ad_best[k].sstep / fwd_best[k].sstep for k in ks]
    isempty(xs) && continue
    scatterlines!(ax, xs, ys;
                  color = get(bcolor, b, :gray30),
                  marker = get(bmarker, b, :xcross), markersize = 11,
                  linewidth = 2, label = b)
end

hlines!(ax, [1.0]; color = (:gray, 0.6), linestyle = :dash, label = "1× (primal cost)")
axislegend(ax; position = :lc, framevisible = true, labelsize = 11)

save(RATIO_PNG, fig)
println("Wrote $RATIO_PNG")
