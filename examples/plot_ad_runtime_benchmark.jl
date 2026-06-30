# Plot AD runtime benchmark results
# =================================
# Reads ad_runtime_benchmark.csv (produced by ad_runtime_benchmark.jl) and renders
# the checkpointed-AD runtime scaling for the barotropic-gyre and inertial-gravity
# wave cases, comparing the GPU and CPU backends, with CairoMakie. Produces
# ad_runtime_benchmark.png with two panels:
#   (left)  total AD runtime vs number of timesteps  (one line per case × backend)
#   (right) GPU speed-up (CPU runtime / GPU runtime) vs number of timesteps
#
# Run with:  julia examples/plot_ad_runtime_benchmark.jl
# This script needs only CairoMakie + DelimitedFiles (not MOKA), so run it in an
# environment that has CairoMakie — e.g. the default global env. If CairoMakie is
# not installed there: julia -e 'using Pkg; Pkg.add("CairoMakie")'.

using CairoMakie
using DelimitedFiles

const CSV = joinpath(@__DIR__, "ad_runtime_benchmark.csv")
const PNG = joinpath(@__DIR__, "ad_runtime_benchmark.png")

isfile(CSV) || error("$CSV not found — run ad_runtime_benchmark.jl first.")

raw, header = readdlm(CSV, ',', header=true)
header = vec(header)
col(name) = raw[:, findfirst(==(name), header)]

cases    = string.(col("case"))
backends = string.(col("backend"))
nsteps   = Float64.(col("nsteps"))
runtime  = Float64.(col("runtime_s"))

uniq_cases    = unique(cases)
uniq_backends = unique(backends)

# Friendlier labels.
pretty = Dict("barotropic_gyre"       => "Barotropic gyre (10 km, dt=1 s)",
              "inertial_gravity_wave" => "Inertial-gravity wave (200 km, dt=300 s)")
label(c) = get(pretty, c, c)

# Consistent colour per case, line style per backend.
palette   = [:dodgerblue3, :darkorange2, :seagreen, :purple]
casecolor = Dict(c => palette[mod1(i, end)] for (i, c) in enumerate(uniq_cases))
bstyle    = Dict("GPU" => :solid, "CPU" => :dash)
bmarker   = Dict("GPU" => :circle, "CPU" => :utriangle)

# Helper: sorted (x, y) for a (case, backend) selection.
function series(c, b)
    idx = (cases .== c) .& (backends .== b)
    x = nsteps[idx]; y = runtime[idx]
    o = sortperm(x)
    return x[o], y[o]
end

fig = Figure(size = (1050, 440))

ax1 = Axis(fig[1, 1];
           title  = "Checkpointed reverse-mode AD runtime",
           xlabel = "number of timesteps differentiated",
           ylabel = "total AD runtime  [s]",
           xscale = log2, yscale = log10)

for c in uniq_cases, b in uniq_backends
    x, y = series(c, b)
    isempty(x) && continue
    scatterlines!(ax1, x, y;
                  color = casecolor[c],
                  linestyle = get(bstyle, b, :solid),
                  marker = get(bmarker, b, :diamond), markersize = 11,
                  linewidth = 2, label = "$(label(c)) — $b")
end
axislegend(ax1; position = :lt, framevisible = true, labelsize = 11)

# --- Speed-up panel: CPU / GPU runtime per case (only if both backends present) -
ax2 = Axis(fig[1, 2];
           title  = "Backend runtime ratio (CPU / GPU)\n>1 ⇒ GPU faster,  <1 ⇒ CPU faster",
           xlabel = "number of timesteps differentiated",
           ylabel = "CPU runtime / GPU runtime  [×]",
           xscale = log2, yscale = log10)

if "GPU" in uniq_backends && "CPU" in uniq_backends
    for c in uniq_cases
        xg, yg = series(c, "GPU")
        xc, yc = series(c, "CPU")
        # align on common nsteps
        common = intersect(xg, xc)
        isempty(common) && continue
        cs = sort(collect(common))
        speedup = [yc[findfirst(==(n), xc)] / yg[findfirst(==(n), xg)] for n in cs]
        scatterlines!(ax2, cs, speedup;
                      color = casecolor[c], marker = :rect, markersize = 11,
                      linewidth = 2, label = label(c))
    end
    hlines!(ax2, [1.0]; color = (:gray, 0.6), linestyle = :dash)
    axislegend(ax2; position = :rt, framevisible = true, labelsize = 11)
else
    text!(ax2, 0.5, 0.5; text = "need both GPU and CPU\nruns for speed-up",
          align = (:center, :center), space = :relative, color = :gray)
end

save(PNG, fig)
println("Wrote $PNG")
