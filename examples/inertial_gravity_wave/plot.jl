# %% Dependencies
using NCDatasets
using CairoMakie
using Statistics
using Dates
using LinearAlgebra
using DataFrames

# %% Exact Solution Struct
"""
Struct to compute the exact solution for the inertial gravity wave test case.
"""
struct ExactSolution
    angleEdge :: Vector{Float64}
    xCell     :: Vector{Float64}
    yCell     :: Vector{Float64}
    xEdge     :: Vector{Float64}
    yEdge     :: Vector{Float64}
    f0        :: Float64
    eta0      :: Float64
    g         :: Float64
    kx        :: Float64
    ky        :: Float64
    omega     :: Float64
end

"""
    ExactSolution(ds::NCDataset) -> ExactSolution

Construct an ExactSolution from an MPAS mesh NCDataset.
"""
function ExactSolution(ds::NCDataset)
    angleEdge = Array(ds["angleEdge"][:])
    xCell     = Array(ds["xCell"][:])
    yCell     = Array(ds["yCell"][:])
    xEdge     = Array(ds["xEdge"][:])
    yEdge     = Array(ds["yEdge"][:])

    bottom_depth = 1000.0
    f0   = 1e-4
    eta0 = 1.0
    lx   = 10000.0
    npx  = 2.0
    npy  = 2.0
    g    = 9.80616

    ly    = sqrt(3.0) / 2.0 * lx
    kx    = npx * 2.0 * π / (lx * 1e3)
    ky    = npy * 2.0 * π / (ly * 1e3)
    omega = sqrt(f0^2 + g * bottom_depth * (kx^2 + ky^2))

    return ExactSolution(angleEdge, 
    xCell, yCell, xEdge, yEdge,
                         f0, eta0, g, kx, ky, omega)
end

# %% Exact Solution Methods
"""
    ssh(exact::ExactSolution, t::Float64) -> Vector{Float64}

Exact sea surface height solution on cells at time `t`.
"""
function ssh(exact::ExactSolution, t::Float64)
    return exact.eta0 .* cos.(
        exact.kx .* exact.xCell .+
        exact.ky .* exact.yCell .-
        exact.omega * t
    )
end

"""
    normal_velocity(exact::ExactSolution, t::Float64) -> Vector{Float64}

Exact normal velocity solution on edges at time `t`.
"""
function normal_velocity(exact::ExactSolution, t::Float64)
    phase = exact.kx .* exact.xEdge .+ exact.ky .* exact.yEdge .- exact.omega * t

    coeff = exact.g / (exact.omega^2 - exact.f0^2)

    u = exact.eta0 .* coeff .* (
        exact.omega .* exact.kx .* cos.(phase) .-
        exact.f0   .* exact.ky .* sin.(phase)
    )

    v = exact.eta0 .* coeff .* (
        exact.omega .* exact.ky .* cos.(phase) .+
        exact.f0   .* exact.kx .* sin.(phase)
    )

    return u .* cos.(exact.angleEdge) .+ v .* sin.(exact.angleEdge)
end

# %% MPAS / MOJO Preprocessing
const MPAS_DT_fmt = dateformat"yyyy-mm-dd_HH:MM:SS"

"""
    preprocess_MPAS(ds::NCDataset) -> (dt, T_f, ssh_num, vel_num)

Parse MPAS output: extract timestep, final time, SSH, and normal velocity.
"""
function preprocess_MPAS(ds::NCDataset)
    # Read xtime as array of strings, stripping whitespace
    xtime_raw = ds["xtime"][:]
    time_strs = [strip(String(collect(s))) for s in eachcol(xtime_raw)]

    times = [DateTime(s, MPAS_DT_fmt) for s in time_strs]
    T_f   = Float64(Dates.value(times[end] - times[1])) / 1000.0  # ms -> s

    # ssh is (nCells,), normalVelocity is (nEdges, nVertLevels)
    ssh_num = Array(ds["ssh"][:])
    vel_num = Array(ds["normalVelocity"][:, 1])   # take surface level

    dt_str = split(ds.attrib["config_dt"], "_")

    if parse(Int, dt_str[1]) == 0
        t  = Time(dt_str[2], dateformat"HH:MM:SS")
        dt = Float64(
            Dates.value(
                Millisecond(Hour(t) + Minute(t) + Second(t))
            )
        ) / 1000.0
    else
        error("Problem parsing: " * join(dt_str, "_"))
    end

    return dt, T_f, ssh_num, vel_num
end

"""
    preprocess_MOJO(ds::NCDataset) -> (dt, T_f, ssh_num, vel_num)

Parse MOJO output: extract timestep, final time, SSH, and normal velocity.
"""
function preprocess_MOJO(ds::NCDataset)
    # time is a variable, not an attribute
    T_f = Float64(Array(ds["time"][:])[end])

    # dt is a global attribute
    dt = Float64(ds.attrib["dt"])

    # ssh is (nCells,) — no time dimension
    ssh_num = Array(ds["ssh"][:])

    # normalVelocity is (nEdges × nVertLevels), take surface level
    vel_num = Array(ds["normalVelocity"][:, 1])

    return dt, T_f, ssh_num, vel_num
end


"""
    read_and_compare(ds_fp, mesh_fp) -> (results::Dict, mesh_ds::NCDataset)

Load numerical results, compute analytical solution, and return comparison dict.
"""
function read_and_compare(ds_fp::String, mesh_fp::String)

    num_ds = NCDataset(ds_fp)

    if haskey(num_ds.attrib, "model_name")
        mesh_ds = NCDataset(mesh_fp)
        exact   = ExactSolution(mesh_ds)
        dt, T_f, ssh_num, vel_num = preprocess_MPAS(num_ds)
        dcEdge  = mean(Array(mesh_ds["dcEdge"][:])) / 1e3
        close(mesh_ds)
    else
        mesh_ds = NCDataset(mesh_fp)
        exact   = ExactSolution(mesh_ds)
        dt, T_f, ssh_num, vel_num = preprocess_MOJO(num_ds)
        dcEdge  = mean(Array(num_ds["dcEdge"][:])) / 1e3
    end

    ssh_ext = ssh(exact, T_f)
    vel_ext = normal_velocity(exact, T_f)

    results = Dict(
        "ssh_num" => ssh_num,
        "vel_num" => vel_num,
        "ssh_ext" => ssh_ext,
        "vel_ext" => vel_ext,
        "ssh_err" => ssh_ext .- ssh_num,
        "vel_err" => vel_ext .- vel_num,
        "dt"      => dt,
        "time"    => T_f,
        "dcEdge"  => dcEdge,
    )

    close(num_ds)

    # Return both results Dict and the ExactSolution for optional plotting
    return results, exact
end

# %% Error Logging
"""
    log_error(results::Dict) -> DataFrame

Compute RMSE for SSH and return as a one-row DataFrame.
"""
function log_error(results::Dict)
    T_f = results["time"]
    dc  = results["dcEdge"]
    dt  = results["dt"]

    ssh_error = sqrt(mean(results["ssh_err"] .^ 2))

    return DataFrame(
        T_f      = T_f,
        dt       = dt,
        dc       = dc,
        rmse_ssh = ssh_error,
    )
end

# %% Plot Fields
function plot_fields(results::Dict, exact::ExactSolution, res_num::String="")
    fig = Figure(size = (1200, 800))
    
    ax1 = Axis(fig[1, 1], title="Numerical SSH, Res: $(res_num)")
    sc1 = scatter!(ax1, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_num"], colormap=:RdBu, markersize=5)
    Colorbar(fig[1, 2], sc1)

    ax2 = Axis(fig[1, 3], title="Exact SSH")
    sc2 = scatter!(ax2, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_ext"], colormap=:RdBu, markersize=5)
    Colorbar(fig[1, 4], sc2)

    ax3 = Axis(fig[1, 5], title="Error SSH")
    sc3 = scatter!(ax3, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_err"], colormap=:viridis, colorrange=(-maximum(abs.(results["ssh_err"])), maximum(abs.(results["ssh_err"]))), markersize=5)
    Colorbar(fig[1, 6], sc3)

    ax4 = Axis(fig[2, 1], title="Numerical Normal Velocity")
    sc4 = scatter!(ax4, exact.xEdge ./ 1e3, exact.yEdge ./ 1e3, color=results["vel_num"], colormap=:RdBu, markersize=5)
    Colorbar(fig[2, 2], sc4)

    ax5 = Axis(fig[2, 3], title="Exact Normal Velocity")
    sc5 = scatter!(ax5, exact.xEdge ./ 1e3, exact.yEdge ./ 1e3, color=results["vel_ext"], colormap=:RdBu, markersize=5)
    Colorbar(fig[2, 4], sc5)   

    ax6 = Axis(fig[2, 5], title="Error Normal Velocity")
    sc6 = scatter!(ax6, exact.xEdge ./ 1e3, exact.yEdge ./ 1e3, color=results["vel_err"], colormap=:viridis , colorrange=(-maximum(abs.(results["vel_err"])), maximum(abs.(results["vel_err"]))), markersize=5)
    Colorbar(fig[2, 6], sc6)
    
    # save("ssh_comparison_$(res_name).png", fig; px_per_unit = 2)
    return fig
end

# %% Main Script: Convergence Plot
# Usually runs on different resolutions like 240km, 120km, 60km, 30km or something equivalent for gyre. 
# We'll put generic folders here, adjusting based on actual simulated data.

# res_dirs = ["40km", "20km", "10km"]

# for dir_ in reverse(res_dirs)
res = "25km"
dir_ = joinpath(@__DIR__, res)
mesh_fp = joinpath(dir_, "initial_state.nc")
out_fp = joinpath(dir_, "output.nc")

results, exact = read_and_compare(out_fp, mesh_fp)
# push!(df_list, log_error(results))
fig = plot_fields(results, exact, res)
display(fig)

# %%
xCell = exact.xCell
yCell = exact.yCell
res_num = reshape(results["ssh_num"], )