# Plot per-kernel CPU-vs-GPU comparison (barotropic gyre)
# =======================================================
# Reads kernel_benchmark.csv (written by kernel_benchmark.jl) and renders, for EACH
# individual forward-step kernel, how its runtime compares ACROSS backends — the CPU
# and every GPU vendor present (NVIDIA CUDA, AMD ROCm, Intel oneAPI) — as mesh
# resolution grows, with CairoMakie.
#
# Layout: a grid of small-multiple panels, ONE PER KERNEL. Within each panel every
# backend is a separate line (min runtime per launch vs cell count N, log-log), so the
# CPU/GPU gap for that specific kernel — and how it widens or narrows with resolution —
# is read off directly. A single shared legend labels the backends for the whole figure.
#
# Composite drivers (diagnostic_compute!, ocn_timestep, the two compute*Tendency!) are
# excluded so the grid shows only the primitive kernels (their parts) without double
# counting. Set KBENCH_COMPOSITE=1 to include them as extra panels.
#
# Fonts are sized for print (a two-column paper figure): pass KBENCH_FONTSCALE to scale
# every text element up or down uniformly (default 1.0).
#
# Run with:  julia examples/barotropic_gyre/plot_kernel_benchmark.jl
# Needs only CairoMakie + DelimitedFiles (not UnstructuredOceans) — run it in an environment that has
# CairoMakie (e.g. the global env; add with Pkg.add("CairoMakie")).
#
# Environment overrides (all optional):
#   KBENCH_CSV=/path/in.csv    # input CSV       (default: kernel_benchmark.csv here)
#   KBENCH_PNG=/path/out.png   # output figure   (default: kernel_benchmark.png here)
#   KBENCH_COMPOSITE=1         # also plot the composite-driver panels (default: off)
#   KBENCH_NCOLS=2             # panels per row  (default: 2)
#   KBENCH_FONTSCALE=1.2       # scale all fonts (default: 1.0)
#   KBENCH_PLOT_DEVICE=<str>   # pin one physical device string per backend label

using CairoMakie
using DelimitedFiles
using Printf

const CSV = get(ENV, "KBENCH_CSV", joinpath(@__DIR__, "kernel_benchmark.csv"))
const PNG = get(ENV, "KBENCH_PNG", joinpath(@__DIR__, "kernel_benchmark.png"))

isfile(CSV) || error("$CSV not found — run kernel_benchmark.jl first.")

raw, header = readdlm(CSV, ',', header=true)
header = vec(header)
col(name) = raw[:, findfirst(==(name), header)]

backend   = string.(col("backend"))
group     = string.(col("group"))
kernel    = string.(col("kernel"))
ncells    = Float64.(col("nCells"))
min_ns    = Float64.(col("min_ns"))
# `device` is present in the current schema; older CSVs lack it → "unknown".
device    = "device" in header ? string.(col("device")) : fill("unknown", size(raw, 1))

# Per-kernel panels: drop the composite drivers unless explicitly requested (their
# constituent kernels are plotted individually, so keeping them double-counts).
const SHOW_COMPOSITE = get(ENV, "KBENCH_COMPOSITE", "0") == "1"
keep = SHOW_COMPOSITE ? trues(length(group)) : (group .!= "composite")
backend = backend[keep]; group = group[keep]; kernel = kernel[keep]
ncells  = ncells[keep];  min_ns = min_ns[keep]; device = device[keep]

uniq_kernels = unique(kernel)                    # stable first-seen (pipeline) order

# Backend ordering + styling. The CPU is the baseline (dashed, warm) so the GPU-vs-CPU
# gap reads at a glance; GPU vendors are solid with distinct cool/vendor colors. Any
# backend not in these maps falls back to a gray solid line so an unknown vendor still
# plots.
const BORDER  = ["CPU", "CUDA", "AMD", "oneAPI"]
bcolor  = Dict("CPU" => :darkorange2, "CUDA" => :seagreen4,
               "AMD" => :firebrick3, "oneAPI" => :dodgerblue3)
bstyle  = Dict("CPU" => :dash, "CUDA" => :solid, "AMD" => :solid, "oneAPI" => :solid)
bmarker = Dict("CPU" => :utriangle, "CUDA" => :circle,
               "AMD" => :diamond, "oneAPI" => :rect)

# Backends present, in the canonical CPU→GPU order, with any extras appended.
present = unique(backend)
uniq_backends = vcat(filter(b -> b in present, BORDER),
                     filter(b -> !(b in BORDER), present))

# Sorted (x, y) for one kernel on one backend, restricted to finite y. The CSV is
# append-mode and may hold several rows for the same (kernel, backend, ncells) — repeat
# runs, or multiple HPC nodes sharing a backend name — so collapse duplicates to the MIN
# value (the standard low-noise estimator), one point per cell count.
#
# Device-aware: a backend label can span physically different cards. We still collapse to
# one point per cell count but WARN when it spans distinct devices, so mixed hardware is
# never mistaken for one machine. Pin a single card with KBENCH_PLOT_DEVICE=<string>.
const PLOT_DEVICE = get(ENV, "KBENCH_PLOT_DEVICE", "")

function series(k, b)
    idx = (kernel .== k) .& (backend .== b) .& isfinite.(min_ns)
    isempty(PLOT_DEVICE) || (idx = idx .& (device .== PLOT_DEVICE))
    x = ncells[idx]; y = min_ns[idx]; dev = device[idx]
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

# --- Print a compact CPU→GPU speed-up summary at the largest common resolution -------
# For each kernel, report how many × faster the fastest GPU is than the CPU at the
# largest cell count both measured, so the figure has a quotable companion table.
function speedup_summary()
    "CPU" in uniq_backends || return
    gpus = filter(!=("CPU"), uniq_backends)
    isempty(gpus) && return
    println("\nCPU→GPU speed-up per kernel (min runtime, largest common N):")
    @printf("  %-22s %10s %10s  %s\n", "kernel", "N", "speedup", "fastest GPU")
    for k in uniq_kernels
        xc, yc = series(k, "CPU")
        isempty(xc) && continue
        best_ratio = -Inf; best_N = NaN; best_gpu = ""
        for g in gpus
            xg, yg = series(k, g)
            common = intersect(xc, xg)
            isempty(common) && continue
            N = maximum(common)
            r = yc[findfirst(==(N), xc)] / yg[findfirst(==(N), xg)]
            if r > best_ratio
                best_ratio, best_N, best_gpu = r, N, g
            end
        end
        isfinite(best_ratio) &&
            @printf("  %-22s %10d %9.1f× %s\n", k, Int(best_N), best_ratio, best_gpu)
    end
end

# --- Figure ------------------------------------------------------------------------
# Paper-sized type: a single FONTSCALE knob scales every text element together so the
# figure stays legible when shrunk into a two-column layout.
const FS      = parse(Float64, get(ENV, "KBENCH_FONTSCALE", "1.0"))
const NCOLS   = parse(Int, get(ENV, "KBENCH_NCOLS", "2"))
const nkerns  = length(uniq_kernels)
const nrows   = cld(nkerns, NCOLS)

const BASEFONT   = 22 * FS
const TITLEFONT  = 24 * FS
const LABELFONT  = 22 * FS
const TICKFONT   = 17 * FS
const LEGENDFONT = 23 * FS

fig = Figure(size = (430 * NCOLS, 380 * nrows + 130), fontsize = BASEFONT)

for (i, k) in enumerate(uniq_kernels)
    r = cld(i, NCOLS); c = i - (r - 1) * NCOLS
    # Only the bottom row of panels gets an x-label / the left column a y-label, so the
    # shared axis text isn't repeated across every small multiple (paper-clean).
    bottom = (r == nrows) || (i + NCOLS > nkerns)
    ax = Axis(fig[r, c];
              title       = k,
              titlesize   = TITLEFONT,
              xlabel      = bottom ? "number of cells  N" : "",
              ylabel      = c == 1 ? "min runtime / launch  [µs]" : "",
              xlabelsize  = LABELFONT, ylabelsize = LABELFONT,
              xticklabelsize = TICKFONT, yticklabelsize = TICKFONT,
              xscale = log10, yscale = log10)

    for b in uniq_backends
        x, y = series(k, b)
        isempty(x) && continue
        scatterlines!(ax, x, y ./ 1e3;                 # ns → µs
                      color     = get(bcolor, b, :gray30),
                      linestyle = get(bstyle, b, :solid),
                      marker    = get(bmarker, b, :xcross),
                      markersize = 15 * FS, linewidth = 3 * FS)
    end
end

# One shared legend for the whole figure, spanning the row below the grid.
legend_elems = [[LineElement(color = get(bcolor, b, :gray30),
                             linestyle = get(bstyle, b, :solid), linewidth = 3 * FS),
                 MarkerElement(color = get(bcolor, b, :gray30),
                               marker = get(bmarker, b, :xcross), markersize = 16 * FS)]
                for b in uniq_backends]
Legend(fig[nrows + 1, 1:NCOLS], legend_elems, uniq_backends, "Backend";
       orientation = :horizontal, framevisible = true,
       labelsize = LEGENDFONT, titlesize = LEGENDFONT,
       titlefont = :bold, patchsize = (40 * FS, 20 * FS), colgap = 24 * FS)

Label(fig[0, 1:NCOLS], "Per-kernel runtime: CPU vs GPU (barotropic gyre)";
      fontsize = TITLEFONT + 4 * FS, font = :bold, padding = (0, 0, 8, 0))

save(PNG, fig)
@printf("Wrote %s\n", PNG)
speedup_summary()
