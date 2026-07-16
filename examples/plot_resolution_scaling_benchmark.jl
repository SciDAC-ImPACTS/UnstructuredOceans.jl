# Plot resolution-scaling benchmark results
# =========================================
# Reads the forward CSV (resolution_scaling_benchmark.csv) and, if present, the AD
# CSV (resolution_scaling_ad_benchmark.csv) — written by the two now-separate
# benchmark scripts — and renders how runtime scales with mesh size (cell count),
# comparing the GPU and CPU backends, with CairoMakie.
#
# The forward and AD results are plotted on SEPARATE figures (they are different
# drivers with different step counts, so putting them on one graph invites
# apples-to-oranges reading):
#   resolution_scaling_benchmark.png    — forward model
#   resolution_scaling_ad_benchmark.png — checkpointed AD (reverse)
#
# Each figure has two panels:
#   (left)  runtime per timestep vs cell count  (one line per backend)
#   (right) GPU speed-up (CPU s/step / GPU s/step) vs cell count
#
# The per-step runtime (not total) is plotted so backends are comparable. A
# reference O(N) slope is drawn so you can read off how close each curve is to
# linear scaling in N.
#
# Run with:  julia examples/plot_resolution_scaling_benchmark.jl
# Needs only CairoMakie + DelimitedFiles (not MOKA) — run it in an environment
# that has CairoMakie (e.g. the global env; add with Pkg.add("CairoMakie")).

using CairoMakie
using DelimitedFiles

const FWD_CSV = joinpath(@__DIR__, "resolution_scaling_benchmark.csv")
const AD_CSV  = joinpath(@__DIR__, "resolution_scaling_ad_benchmark.csv")
const FWD_PNG = joinpath(@__DIR__, "resolution_scaling_benchmark.png")
const AD_PNG  = joinpath(@__DIR__, "resolution_scaling_ad_benchmark.png")

# Pull the columns we plot out of one CSV; returns empty vectors if it is absent.
function load_csv(path)
    isfile(path) || return (backend = String[], mode = String[],
                            ncells = Float64[], sstep = Float64[], problem = String[])
    raw, header = readdlm(path, ',', header=true)
    header = vec(header)
    col(name) = raw[:, findfirst(==(name), header)]
    # CSVs written before the `problem` column was added default to the IGW.
    problem = "problem" in header ? string.(col("problem")) : fill("igw", size(raw, 1))
    return (backend = string.(col("backend")), mode = string.(col("mode")),
            ncells = Float64.(col("ncells")), sstep = Float64.(col("s_per_step")),
            problem = problem)
end

fwd = load_csv(FWD_CSV)
ad  = load_csv(AD_CSV)
isempty(fwd.mode) && isempty(ad.mode) &&
    error("neither $FWD_CSV nor $AD_CSV found — run the benchmark script(s) first.")

# Human-readable problem name for the title.
probtitle = Dict("gyre" => "barotropic gyre",
                 "igw"  => "inertial-gravity wave")
probname_of(problem) = get(probtitle, problem, problem)

bstyle    = Dict("GPU" => :solid, "CPU" => :dash)
bmarker   = Dict("GPU" => :circle, "CPU" => :utriangle)
bcolor    = Dict("GPU" => :dodgerblue3, "CPU" => :darkorange2)

# Helper: sorted (x, y) for a backend selection within one dataset.
function series(data, b)
    idx = data.backend .== b
    x = data.ncells[idx]; y = data.sstep[idx]
    o = sortperm(x)
    return x[o], y[o]
end

# Render one figure (runtime + speed-up panels) for a single dataset (forward or AD).
function plot_dataset(data, label, png)
    isempty(data.mode) && (println("skip $label — no data"); return)

    probname = probname_of(first(data.problem))
    uniq_backends = unique(data.backend)

    fig = Figure(size = (1050, 440))

    ax1 = Axis(fig[1, 1];
               title  = "$label runtime scaling ($probname)",
               xlabel = "number of cells  N",
               ylabel = "runtime per timestep  [s]",
               xscale = log10, yscale = log10)

    for b in uniq_backends
        x, y = series(data, b)
        isempty(x) && continue
        scatterlines!(ax1, x, y;
                      color = get(bcolor, b, :black),
                      linestyle = get(bstyle, b, :solid),
                      marker = get(bmarker, b, :diamond), markersize = 11,
                      linewidth = 2, label = b)
    end

    # Reference O(N) slope anchored at the smallest measured point.
    if !isempty(data.ncells)
        x0 = minimum(data.ncells)
        idx0 = findfirst(==(x0), data.ncells)
        y0 = data.sstep[idx0]
        xr = [x0, maximum(data.ncells)]
        lines!(ax1, xr, y0 .* (xr ./ x0); color = (:gray, 0.6), linestyle = :dot,
               linewidth = 2, label = "O(N) reference")
    end
    axislegend(ax1; position = :lt, framevisible = true, labelsize = 11)

    # --- Speed-up panel: CPU / GPU per-step runtime -------------------------------
    ax2 = Axis(fig[1, 2];
               title  = "Backend runtime ratio (CPU / GPU)\n>1 ⇒ GPU faster",
               xlabel = "number of cells  N",
               ylabel = "CPU s/step / GPU s/step  [×]",
               xscale = log10, yscale = log10)

    if "GPU" in uniq_backends && "CPU" in uniq_backends
        xg, yg = series(data, "GPU")
        xc, yc = series(data, "CPU")
        common = intersect(xg, xc)
        if !isempty(common)
            cs = sort(collect(common))
            speedup = [yc[findfirst(==(n), xc)] / yg[findfirst(==(n), xg)] for n in cs]
            scatterlines!(ax2, cs, speedup;
                          color = :seagreen, marker = :rect, markersize = 11,
                          linewidth = 2)
            hlines!(ax2, [1.0]; color = (:gray, 0.6), linestyle = :dash)
        end
    else
        text!(ax2, 0.5, 0.5; text = "need both GPU and CPU\nruns for speed-up",
              align = (:center, :center), space = :relative, color = :gray)
    end

    save(png, fig)
    println("Wrote $png")
end

plot_dataset(fwd, "Forward",       FWD_PNG)
plot_dataset(ad,  "AD (reverse)",  AD_PNG)
