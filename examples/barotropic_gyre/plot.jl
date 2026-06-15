# %% Dependencies
using NCDatasets
using CairoMakie

# %% Read fields from a MOKA output / initial-state pair
"""
    read_fields(out_fp, mesh_fp) -> Dict

Read cell/edge coordinates plus the surface SSH, normal velocity, and the
zonal wind-stress forcing for plotting.
"""
function read_fields(out_fp::String, mesh_fp::String)
    mesh_ds = NCDataset(mesh_fp)
    num_ds  = NCDataset(out_fp)

    fields = Dict(
        "xCell" => Array(mesh_ds["xCell"][:]),
        "yCell" => Array(mesh_ds["yCell"][:]),
        "xEdge" => Array(mesh_ds["xEdge"][:]),
        "yEdge" => Array(mesh_ds["yEdge"][:]),
        "ssh"   => Array(num_ds["ssh"][:]),
        "vel"   => Array(num_ds["normalVelocity"][:, 1]),
        "tau"   => Array(mesh_ds["windStressZonal"][:]),
    )

    close(num_ds)
    close(mesh_ds)
    return fields
end

# %% Plot SSH, normal velocity, and wind-stress forcing
function plot_fields(f::Dict, res_num::String="")
    xc = f["xCell"] ./ 1e3; yc = f["yCell"] ./ 1e3
    xe = f["xEdge"] ./ 1e3; ye = f["yEdge"] ./ 1e3

    fig = Figure(size = (1200, 380))

    ax1 = Axis(fig[1, 1], title="SSH, Res: $(res_num)",
               xlabel="x (km)", ylabel="y (km)", aspect=1)
    s1 = scatter!(ax1, xc, yc, color=f["ssh"], colormap=:balance, markersize=5)
    Colorbar(fig[1, 2], s1)

    ax2 = Axis(fig[1, 3], title="Normal velocity",
               xlabel="x (km)", aspect=1)
    s2 = scatter!(ax2, xe, ye, color=f["vel"], colormap=:balance, markersize=5)
    Colorbar(fig[1, 4], s2)

    ax3 = Axis(fig[1, 5], title="Zonal wind stress",
               xlabel="x (km)", aspect=1)
    s3 = scatter!(ax3, xc, yc, color=f["tau"], colormap=:balance, markersize=5)
    Colorbar(fig[1, 6], s3)

    return fig
end

# %% Main
res     = "10km"
dir_    = joinpath(@__DIR__, res)
mesh_fp = joinpath(dir_, "initial_state.nc")
out_fp  = joinpath(dir_, "output.nc")

fields = read_fields(out_fp, mesh_fp)
fig = plot_fields(fields, res)
save(joinpath(@__DIR__, "fields.png"), fig; px_per_unit = 3)
display(fig)
