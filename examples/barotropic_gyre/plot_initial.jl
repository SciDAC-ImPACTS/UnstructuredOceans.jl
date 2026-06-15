using NCDatasets
using CairoMakie

function read_initial(dir::String)
    ds = NCDataset(joinpath(dir, "initial_state.nc"))

    xc    = Array(ds["xCell"][:]) ./ 1e3   # → km
    yc    = Array(ds["yCell"][:]) ./ 1e3
    tau_x  = vec(Array(ds["windStressZonal"][:]))
    f_cell = vec(Array(ds["fCell"][:]))

    close(ds)
    return (; xc, yc, tau_x, f_cell)
end

# Sort by y for smooth 1-D profile lines.
function ysorted(y, v)
    idx = sortperm(y)
    return y[idx], v[idx]
end

function plot_initial(dir::String, res::String="")
    f = read_initial(dir)
    xc, yc = f.xc, f.yc

    fig = Figure(size = (1400, 700))

    label = isempty(res) ? "" : "  [$res]"
    Label(fig[0, :]; text = "Barotropic gyre — initial state & forcing$label",
          fontsize = 16, font = :bold)

    # ── Zonal wind stress ──────────────────────────────────────────────────
    ax1 = Axis(fig[1, 1]; title = "Zonal wind stress τₓ (N m⁻²)",
               xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
    s1 = scatter!(ax1, xc, yc; color = f.tau_x, colormap = :RdBu, markersize = 5)
    Colorbar(fig[1, 2], s1)

    ys, ts = ysorted(yc, f.tau_x)
    ax2 = Axis(fig[2, 1:2]; title = "τₓ meridional profile",
               xlabel = "τₓ (N m⁻²)", ylabel = "y (km)")
    lines!(ax2, ts, ys; color = :steelblue, linewidth = 2)
    vlines!(ax2, [0.0]; color = :black, linestyle = :dash, linewidth = 1)

    # ── Beta-plane Coriolis ────────────────────────────────────────────────
    ax3 = Axis(fig[1, 3]; title = "Beta-plane f (s⁻¹)",
               xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
    s3 = scatter!(ax3, xc, yc; color = f.f_cell, colormap = :viridis, markersize = 5)
    Colorbar(fig[1, 4], s3)

    ys, fs = ysorted(yc, f.f_cell)
    ax4 = Axis(fig[2, 3:4]; title = "f meridional profile",
               xlabel = "f (s⁻¹)", ylabel = "y (km)")
    lines!(ax4, fs, ys; color = :darkorange, linewidth = 2)

    return fig
end

res = "10km"
dir = joinpath(@__DIR__, res)

fig = plot_initial(dir, res)
outpath = joinpath(@__DIR__, "initial_fields.png")
save(outpath, fig; px_per_unit = 3)
display(fig)
println("Saved → $outpath")
