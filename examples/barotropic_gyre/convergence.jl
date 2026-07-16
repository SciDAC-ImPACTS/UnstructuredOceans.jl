# %% Dependencies
# Reuse the Munk exact solution, numerical streamfunction reconstruction, and the
# area-weighted L2 error from analysis.jl (included as a library — its main block
# is guarded, so nothing runs on include).
include("analysis.jl")

using DataFrames

# %% Main Script: Convergence Plot
# Spatial convergence of the barotropic (Munk) gyre: each resolution is spun up to
# the same 20-day quasi-steady state and its reconstructed streamfunction compared
# against the analytical Munk solution. Only resolutions that resolve the Munk
# boundary layer (~58 km wide, see setup.py) sit in the asymptotic regime.
resolutions = ["5km", "10km", "20km"]

df_list = DataFrame[]

for dir_ in resolutions
    mesh_fp   = joinpath(@__DIR__, dir_, "initial_state.nc")
    output_fp = joinpath(@__DIR__, dir_, "output.nc")

    results, mesh_ds = read_and_compare(output_fp, mesh_fp)
    dc = mean(Array(mesh_ds["dcEdge"][:])) / 1e3   # mean cell spacing [km]
    close(mesh_ds)

    push!(df_list, DataFrame(dc = dc, l2_err = results["l2_err"]))
end

df = vcat(df_list...)
sort!(df, :dc)   # ascending resolution for a clean fit / plot

# --- Polynomial fit in log10 space ---
log_dc  = log10.(df.dc)
log_err = log10.(df.l2_err)

# polyfit: degree 1 => [slope, intercept]
A    = hcat(log_dc, ones(length(log_dc)))
poly = A \ log_err          # least-squares solution

convergence = poly[1]
conv_round  = round(convergence; digits=3)

fit    = df.dc .^ poly[1] .* 10 .^ poly[2]
order1 = 0.5 .* df.l2_err[end] .* (df.dc ./ df.dc[end])
order2 = 0.5 .* df.l2_err[end] .* (df.dc ./ df.dc[end]) .^ 2

# --- CairoMakie convergence plot ---
fig = Figure(size = (500, 400))
ax  = Axis(fig[1, 1],
    xscale = log10,
    yscale = log10,
    title  = "Barotropic Gyre (Munk, $(BC))",
    xlabel = "Resolution (km)",
    ylabel = "Streamfunction L2 error",
)

lines!(ax, df.dc, order1;
    color     = :black,
    linestyle = :dash,
    alpha     = 0.3,
    label     = "First order",
)

lines!(ax, df.dc, order2;
    color     = :black,
    linestyle = :solid,
    alpha     = 0.3,
    label     = "Second order",
)

lines!(ax, df.dc, fit;
    color = :black,
    label = "Linear fit (order = $(conv_round))",
)

scatter!(ax, df.dc, df.l2_err;
    marker = :circle,
    label  = "L2 error",
)

axislegend(ax; position = :lt)

save(joinpath(@__DIR__, "convergence.png"), fig; px_per_unit = 3)   # ~300 dpi
display(fig)

println("Barotropic gyre streamfunction convergence order: $(conv_round)")
