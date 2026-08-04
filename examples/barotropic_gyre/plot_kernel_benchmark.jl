# Plot per-kernel benchmark results (barotropic gyre)
# ===================================================
# Reads kernel_benchmark.csv (written by kernel_benchmark.jl) and renders, per
# backend, how each individual forward-step kernel scales with mesh resolution, with
# CairoMakie. Two panels per backend row:
#
#   (left)  min runtime per launch  vs  cell count N    — absolute cost; an O(N)
#           reference slope is drawn so super-linear kernels are obvious.
#   (right) time per element (min_ns / extent)  vs  N   — the suboptimality diagnostic:
#           a FLAT line is perfect linear scaling; a RISING line means the kernel does
#           more-than-linear work per element as the mesh grows (the thing to fix).
#
# Composite drivers (diagnostic_compute!, ocn_timestep, the two compute*Tendency!) are
# excluded from the per-kernel lines to avoid double-counting; the individual kernels
# they wrap are plotted instead. The kernel with the worst per-element scaling is
# highlighted (thicker line + explicit legend entry).
#
# Run with:  julia examples/barotropic_gyre/plot_kernel_benchmark.jl
# Needs only CairoMakie + DelimitedFiles (not MOKA) — run it in an environment that has
# CairoMakie (e.g. the global env; add with Pkg.add("CairoMakie")).

using CairoMakie
using DelimitedFiles
using Printf

const CSV = joinpath(@__DIR__, "kernel_benchmark.csv")
const PNG = joinpath(@__DIR__, "kernel_benchmark.png")

isfile(CSV) || error("$CSV not found — run kernel_benchmark.jl first.")

raw, header = readdlm(CSV, ',', header=true)
header = vec(header)
col(name) = raw[:, findfirst(==(name), header)]

backend   = string.(col("backend"))
group     = string.(col("group"))
kernel    = string.(col("kernel"))
ncells    = Float64.(col("nCells"))
extcount  = Float64.(col("extent_count"))
min_ns    = Float64.(col("min_ns"))
ns_elem   = Float64.(col("ns_per_elem"))     # min_ns / extent_count (NaN when extent unknown)
# `device` is present in the current schema; older CSVs lack it → "unknown".
device    = "device" in header ? string.(col("device")) : fill("unknown", size(raw, 1))

# Per-kernel lines only: drop the composite drivers (their parts are plotted separately).
keep       = group .!= "composite"
backend    = backend[keep]; group = group[keep]; kernel = kernel[keep]
ncells     = ncells[keep];  extcount = extcount[keep]
min_ns     = min_ns[keep];  ns_elem = ns_elem[keep]
device     = device[keep]

uniq_backends = unique(backend)
uniq_kernels  = unique(kernel)                       # stable first-seen order
kcolor = Dict(k => c for (k, c) in
              zip(uniq_kernels, Makie.categorical_colors(:tab20, max(length(uniq_kernels), 2))))

# Sorted (x, y) for one kernel on one backend, restricted to finite y. The CSV is
# append-mode and may hold several rows for the same (kernel, backend, ncells) — repeat
# runs, or multiple HPC nodes sharing a backend name — so collapse duplicates to the MIN
# value (the standard low-noise estimator), one point per cell count.
#
# Device-aware: a backend label can span physically different cards. We still collapse to
# one point per cell count but WARN when it spans distinct devices, so mixed hardware is
# never mistaken for one machine. Pin a single card with KBENCH_PLOT_DEVICE=<string>.
const PLOT_DEVICE = get(ENV, "KBENCH_PLOT_DEVICE", "")

function series(vals, k, b)
    idx = (kernel .== k) .& (backend .== b) .& isfinite.(vals)
    isempty(PLOT_DEVICE) || (idx = idx .& (device .== PLOT_DEVICE))
    x = ncells[idx]; y = vals[idx]; dev = device[idx]
    best = Dict{Float64,Float64}()
    devs = Dict{Float64,Set{String}}()
    for (xi, yi, di) in zip(x, y, dev)
        best[xi] = haskey(best, xi) ? min(best[xi], yi) : yi
        push!(get!(devs, xi, Set{String}()), di)
    end
    any(length(s) > 1 for s in values(devs)) && @warn "kernel $k / backend $b: \
        min-collapsed across multiple devices — plotted point is the fastest card, not \
        one machine. Set KBENCH_PLOT_DEVICE to pin one."
    xs = sort(collect(keys(best)))
    return xs, [best[xi] for xi in xs]
end

# Rank kernels by their WORST local scaling exponent: the steepest log-log slope of
# absolute runtime between consecutive meshes (on the first backend). Exponent 1 = linear
# (ideal above the launch-overhead floor); >1 = super-linear = suboptimal. This uses the
# steepest *segment* rather than the endpoints, so a mid-range cliff isn't hidden by an
# overhead-inflated smallest mesh. The worst kernel is highlighted.
function worst_exponent(k, b)
    x, y = series(min_ns, k, b)
    length(x) < 2 && return NaN
    slopes = [log(y[i+1] / y[i]) / log(x[i+1] / x[i]) for i in 1:length(x)-1]
    return maximum(slopes)
end
b0      = first(uniq_backends)
ratios  = [(k, worst_exponent(k, b0)) for k in uniq_kernels]
# Need ≥2 resolutions for a finite scaling exponent; with a single-resolution CSV
# (a partial/just-started multi-node collection) none are finite, so highlight nothing.
finite_ratios = filter(r -> isfinite(r[2]), ratios)
worst  = isempty(finite_ratios) ? "" :
         first(sort(finite_ratios, by = last, rev = true))[1]

lw(k)  = k == worst ? 4 : 2
mz(k)  = k == worst ? 13 : 8

fig = Figure(size = (1150, 430 * length(uniq_backends)))

for (row, b) in enumerate(uniq_backends)
    ax1 = Axis(fig[row, 1];
               title  = "[$b] kernel runtime vs resolution (barotropic gyre)",
               xlabel = "number of cells  N", ylabel = "min runtime per launch  [µs]",
               xscale = log10, yscale = log10)
    ax2 = Axis(fig[row, 2];
               title  = "[$b] time per element  (flat ⇒ linear scaling; rising ⇒ suboptimal)",
               xlabel = "number of cells  N", ylabel = "min_ns / element  [ns]",
               xscale = log10, yscale = log10)

    for k in uniq_kernels
        x1, y1 = series(min_ns, k, b)
        isempty(x1) && continue
        lbl = k == worst ? "$k  ⚠ worst scaling" : k
        scatterlines!(ax1, x1, y1 ./ 1e3; color = kcolor[k],
                      linewidth = lw(k), marker = :circle, markersize = mz(k), label = lbl)

        x2, y2 = series(ns_elem, k, b)
        isempty(x2) || scatterlines!(ax2, x2, y2; color = kcolor[k],
                      linewidth = lw(k), marker = :circle, markersize = mz(k))
    end

    # O(N) reference in the absolute panel, anchored at the smallest measured point.
    finite = isfinite.(min_ns) .& (backend .== b)
    if any(finite)
        x0 = minimum(ncells[finite]); i0 = findfirst(i -> finite[i] && ncells[i] == x0, eachindex(ncells))
        y0 = min_ns[i0] / 1e3; xr = [x0, maximum(ncells[finite])]
        lines!(ax1, xr, y0 .* (xr ./ x0); color = (:gray, 0.55), linestyle = :dot,
               linewidth = 2, label = "O(N) reference")
    end
    axislegend(ax1; position = :lt, framevisible = true, labelsize = 9, nbanks = 2)
end

save(PNG, fig)
@printf("Wrote %s\n", PNG)
isempty(worst) || @printf("Worst scaling kernel: %s (steepest local exponent %.2f on %s; 1.0 = linear)\n",
                          worst, last(first(filter(r -> r[1] == worst, ratios))), b0)
