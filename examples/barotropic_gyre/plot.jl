# %% Dependencies
using NCDatasets
using CairoMakie
using Printf

# %% Read fields from a UnstructuredOceans output / initial-state pair
"""
    read_fields(out_fp, mesh_fp) -> Dict

Read cell/edge coordinates plus the surface SSH, normal velocity, and the
zonal wind-stress forcing for plotting.
"""
function read_fields(out_fp::String, mesh_fp::String; frame::Int=0)
    mesh_ds = NCDataset(mesh_fp)
    num_ds  = NCDataset(out_fp)

    t = frame == 0 ? size(num_ds["ssh"], 2) : frame

    fields = Dict(
        "xCell" => Array(mesh_ds["xCell"][:]),
        "yCell" => Array(mesh_ds["yCell"][:]),
        "xEdge" => Array(mesh_ds["xEdge"][:]),
        "yEdge" => Array(mesh_ds["yEdge"][:]),
        "ssh"   => Array(num_ds["ssh"][:, t]),
        "vel"   => Array(num_ds["normalVelocity"][:, 1, t]),
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

# %% Read all time frames from output
"""
    read_timeseries(out_fp, mesh_fp) -> Dict

Read all checkpoint frames from `output.nc` and return arrays with the time dimension.
`"ssh"` is `(nCells, nT)`, `"vel"` is `(nEdges, nT)`, `"time"` is seconds from start.
"""
function read_timeseries(out_fp::String, mesh_fp::String)
    mesh_ds = NCDataset(mesh_fp)
    num_ds  = NCDataset(out_fp)

    nT = num_ds.dim["time"]

    ts = Dict(
        "xCell" => Array(mesh_ds["xCell"][:]),
        "yCell" => Array(mesh_ds["yCell"][:]),
        "xEdge" => Array(mesh_ds["xEdge"][:]),
        "yEdge" => Array(mesh_ds["yEdge"][:]),
        "tau"   => Array(mesh_ds["windStressZonal"][:]),
        "time"  => Array(num_ds["time"][1:nT]),
        "ssh"   => Array(num_ds["ssh"][:, 1:nT]),
        "vel"   => Array(num_ds["normalVelocity"][:, 1, 1:nT]),
    )

    close(num_ds)
    close(mesh_ds)
    return ts
end

# %% Animation of SSH and normal velocity
"""
    animate_fields(ts, out_path, nT; fps) -> out_path

Record a GIF of SSH and surface normal velocity for all checkpoints in `ts`.
"""
function animate_fields(ts::Dict, out_path::String, nT; fps::Int=8)
    xc = ts["xCell"] ./ 1e3
    yc = ts["yCell"] ./ 1e3
    xe = ts["xEdge"] ./ 1e3
    ye = ts["yEdge"] ./ 1e3
    # nT = length(ts["time"])

    # ssh_lim = max(maximum(abs, ts["ssh"]), 1e-10)
    # vel_lim = max(maximum(abs, ts["vel"]), 1e-10)

    frame_i = Observable(1)
    ssh_obs   = @lift ts["ssh"][:, $frame_i]
    vel_obs   = @lift ts["vel"][:, $frame_i]
    title_obs = @lift @sprintf("t = %.1f days", ts["time"][$frame_i] / 86400.0)

    fig = Figure(size=(800, 600))

    ax1 = Axis(fig[1, 1], title=title_obs,
               xlabel="x (km)", ylabel="y (km)", aspect=1)
    s1 = scatter!(ax1, xc, yc; color=ssh_obs, colormap=:balance,
                #   colorrange=(-ssh_lim, ssh_lim), 
                  markersize=5)
    Colorbar(fig[1, 2], s1; label="SSH (m)")

    ax2 = Axis(fig[2, 1], xlabel="x (km)", aspect=1)
    s2 = scatter!(ax2, xe, ye; color=vel_obs, colormap=:balance,
                #   colorrange=(-vel_lim, vel_lim), 
                  markersize=5)
    Colorbar(fig[2, 2], s2; label="Normal velocity (m/s)")

    record(fig, out_path, 1:nT; framerate=fps) do i
        frame_i[] = i
    end

    return out_path
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

# %% Animate time-series of SSH and normal velocity
ts = read_timeseries(out_fp, mesh_fp)
nT = 26
animate_fields(ts, joinpath(@__DIR__, "animation.mp4"), nT)
