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

    return ExactSolution(angleEdge, xCell, yCell, xEdge, yEdge,
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

    # ssh is (nCells, time) — take last time frame
    ssh_num = Array(ds["ssh"][:, end])

    # normalVelocity is (nEdges × nVertLevels), take surface level
    vel_num = Array(ds["normalVelocity"][:, 1, end])

    return dt, T_f, ssh_num, vel_num
end


# %% Read and Compare
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
        exact   = ExactSolution(num_ds)
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

Compute RMSE for SSH and velocity, return as a one-row DataFrame.
"""
function log_error(results::Dict)
    T_f = results["time"]
    dc  = results["dcEdge"]
    dt  = results["dt"]

    ssh_error = sqrt(mean(results["ssh_err"] .^ 2))
    vel_error = sqrt(mean(results["vel_err"] .^ 2))

    return DataFrame(
        T_f      = T_f,
        dt       = dt,
        dc       = dc,
        rmse_ssh = ssh_error,
        rmse_vel = vel_error,
    )
end

# %% Main Script: Convergence Plot
df_list = DataFrame[]

for dir_ in reverse(["25km", "50km", "100km", "200km"])
    mesh_fp   = joinpath(@__DIR__, dir_, "initial_state.nc")
    output_fp = joinpath(@__DIR__, dir_, "output.nc")

    results, exact = read_and_compare(output_fp, mesh_fp)

    push!(df_list, log_error(results))
end

df = vcat(df_list...)

# --- Polynomial fit in log10 space ---
log_dc  = log10.(df.dc)
log_ssh = log10.(df.rmse_ssh)

# polyfit: degree 1 => [slope, intercept]
A    = hcat(log_dc, ones(length(log_dc)))
poly = A \ log_ssh          # least-squares solution

convergence = poly[1]
conv_round  = round(convergence; digits=3)

fit     = df.dc .^ poly[1] .* 10 .^ poly[2]
order1  = 0.5 .* df.rmse_ssh[end] .* (df.dc ./ df.dc[end])
order2  = 0.5 .* df.rmse_ssh[end] .* (df.dc ./ df.dc[end]) .^ 2

# --- CairoMakie convergence plot ---
fig = Figure(size = (500, 400))
ax  = Axis(fig[1, 1],
    xscale      = log10,
    yscale      = log10,
    title       = "Inertial Gravity Wave",
    xlabel      = "Resolution (km)",
    ylabel      = "SSH RMSE",
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

scatter!(ax, df.dc, df.rmse_ssh;
    marker = :circle,
    label  = "RMSE SSH",
)

axislegend(ax; position = :lt)

save("convergence.png", fig; px_per_unit = 3)   # ~300 dpi equivalent
display(fig)