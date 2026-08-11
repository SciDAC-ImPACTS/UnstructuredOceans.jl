# %% Dependencies
using NCDatasets
using CairoMakie
using Printf

# %% Example cases
# One script, dispatched on the test case. Each case knows where its
# `output_enzyme.nc` lives and how to label itself; everything downstream
# (reading, plotting, saving) is shared.
abstract type ExampleCase end

struct InertialGravityWave <: ExampleCase
    res::String
end
struct BarotropicGyre <: ExampleCase
    res::String
end

InertialGravityWave() = InertialGravityWave("200km")
BarotropicGyre()      = BarotropicGyre("10km")

casedir(c::InertialGravityWave) = joinpath(@__DIR__, "inertial_gravity_wave", c.res)
casedir(c::BarotropicGyre)      = joinpath(@__DIR__, "barotropic_gyre",       c.res)

casename(::InertialGravityWave) = "Inertial Gravity Wave"
casename(::BarotropicGyre)      = "Barotropic Gyre"

resolution(c::ExampleCase) = c.res
output_path(c::ExampleCase) = joinpath(casedir(c), "output_enzyme.nc")

# %% Read primal + adjoint fields
"""
    read_fields(ds_fp; level) -> Dict

Read the primal prognostic fields and their Enzyme-computed adjoints (gradients)
written by `write_netcdf` (the 4-argument method in src/infra/OutPut.jl) from an
`output_enzyme.nc` file.

The `d_*` variables are sensitivities of the seeded scalar objective with respect
to the *initial* prognostic state. Surface (top) vertical level is extracted for
the 3-D fields.

NCDatasets returns dims in reverse (column-major) order, so a variable declared
(time, nVertLevels, nCells) is read back as (nCells, nVertLevels, time).
"""
function read_fields(ds_fp::String; level::Int=1)
    ds = NCDataset(ds_fp)

    fields = Dict{String,Any}(
        # primal
        "ssh"              => Array(ds["ssh"][:, end]),
        "layerThickness"   => Array(ds["layerThickness"][:, level, end]),
        "normalVelocity"   => Array(ds["normalVelocity"][:, level, end]),
        # adjoint / gradient
        "d_ssh"            => Array(ds["d_ssh"][:, end]),
        "d_layerThickness" => Array(ds["d_layerThickness"][:, level, end]),
        "d_normalVelocity" => Array(ds["d_normalVelocity"][:, level, end]),
        # mesh coordinates
        "xCell" => Array(ds["xCell"][:]), "yCell" => Array(ds["yCell"][:]),
        "xEdge" => Array(ds["xEdge"][:]), "yEdge" => Array(ds["yEdge"][:]),
    )

    close(ds)
    return fields
end

"""
    read_mesh_geometry!(fields, mesh_fp) -> Bool

Load the edge→cell reconstruction geometry (`angleEdge`, `edgesOnCell`, `nEdgesOnCell`)
from `mesh_fp` (a case's `initial_state.nc`) into `fields`, returning `true` on success.

The 4-argument enzyme `write_netcdf` (src/infra/OutPut.jl) defines but does NOT populate
`angleEdge`/connectivity in `output_enzyme.nc` (those assignments are commented out), so
they come back as fill values there. The real geometry lives in `initial_state.nc`, which
sits next to the output. Returns `false` (and loads nothing) if that file is absent, so the
caller can fall back to the raw edge scatter.

NCDatasets reads dims reversed, so on-disk `edgesOnCell(nCells, maxEdges)` is returned as
`(maxEdges, nCells)` — indexed `[j, iCell]` below.
"""
function read_mesh_geometry!(fields::Dict, mesh_fp::String)
    isfile(mesh_fp) || return false
    ds = NCDataset(mesh_fp)
    fields["angleEdge"]    = Array(ds["angleEdge"][:])
    fields["edgesOnCell"]  = Array(ds["edgesOnCell"][:, :])
    fields["nEdgesOnCell"] = Array(ds["nEdgesOnCell"][:])
    close(ds)
    return true
end

"""
    reconstruct_cell_velocity(edgevals, angleEdge, edgesOnCell, nEdgesOnCell) -> (u, v)

Reconstruct a cell-centered velocity vector `(u, v)` from the per-edge normal components
`edgevals` (one scalar per edge). Each edge stores `nv_e = u·cos(θ_e) + v·sin(θ_e)` where
`θ_e = angleEdge[e]` is the edge-normal angle to eastward; on the hex mesh θ takes only a
couple of interior values, so coloring the raw per-edge normal alternates sign between
neighbours even for a smooth flow (the salt-and-pepper artifact). Inverting the relation
per cell — a 2×2 least-squares fit over that cell's edges — recovers a smooth field.

The same `u·cosθ + v·sinθ` relation is used (forward) in src/compare.py:128 and
src/inertialGravityWave.jl:61. Cells whose edge set is degenerate (det ≈ 0, e.g. a boundary
cell with too few edges) get `NaN`, which simply renders blank.
"""
function reconstruct_cell_velocity(edgevals, angleEdge, edgesOnCell, nEdgesOnCell)
    nCells = length(nEdgesOnCell)
    u = fill(NaN, nCells)
    v = fill(NaN, nCells)
    for c in 1:nCells
        Scc = Scs = Sss = bx = by = 0.0
        for j in 1:nEdgesOnCell[c]
            e = edgesOnCell[j, c]
            e == 0 && continue
            θ = angleEdge[e]; ct = cos(θ); st = sin(θ)
            Scc += ct*ct; Scs += ct*st; Sss += st*st
            bx  += edgevals[e]*ct; by += edgevals[e]*st
        end
        det = Scc*Sss - Scs*Scs
        if abs(det) > 1e-12
            u[c] = ( Sss*bx - Scs*by) / det
            v[c] = (-Scs*bx + Scc*by) / det
        end
    end
    return u, v
end

# %% Plotting
# One scatter panel + its colorbar, occupying columns (2col-1, 2col) of `row`.
function panel!(fig, row, col, x, y, data, title, cblabel; symmetric::Bool=false)
    kw = symmetric ?
        (colormap = :balance,
         colorrange = (l = max(maximum(abs, data), eps()); (-l, l))) :
        (colormap = :viridis,)

    ax = Axis(fig[row, 2col - 1]; title = title,
              xlabel = "x (km)", ylabel = col == 1 ? "y (km)" : "", aspect = 1)
    s = scatter!(ax, x ./ 1e3, y ./ 1e3; color = data, markersize = 6, kw...)
    Colorbar(fig[row, 2col], s; label = cblabel)
    return ax
end

"""
    plot_fields(f, case) -> Figure

Primal prognostic fields on the first row, their adjoints (∂J/∂·) on the second.
Primal fields use a sequential colormap; adjoints use a diverging colormap
symmetric about zero, since a gradient is signed and the sign carries meaning.
"""
function plot_fields(f::Dict, case::ExampleCase)
    xc, yc = f["xCell"], f["yCell"]
    xe, ye = f["xEdge"], f["yEdge"]
    res    = resolution(case)

    fig = Figure(size = (1200, 1500))

    # Row 1 — primal
    panel!(fig, 1, 1, xc, yc, f["ssh"],            "ssh",            "ssh")
    panel!(fig, 2, 1, xc, yc, f["layerThickness"], "layerThickness", "layerThickness")

    # Row 2 — adjoint / gradient
    panel!(fig, 1, 2, xc, yc, f["d_ssh"],            "∂J/∂ssh",            "adjoint ssh";            symmetric = true)
    panel!(fig, 2, 2, xc, yc, f["d_layerThickness"], "∂J/∂layerThickness", "adjoint layerThickness"; symmetric = true)

    # Column 3 — normalVelocity. The raw per-edge normal component alternates sign between
    # neighbouring edges on the hex mesh (see reconstruct_cell_velocity), so plotting it
    # directly at edge midpoints looks like noise. Reconstruct a cell-centered vector and
    # plot the speed (primal) and the signed zonal component (adjoint) instead. Falls back
    # to the raw edge scatter if the mesh geometry (initial_state.nc) was not found.
    if haskey(f, "angleEdge")
        u, v  = reconstruct_cell_velocity(f["normalVelocity"],   f["angleEdge"], f["edgesOnCell"], f["nEdgesOnCell"])
        du, _ = reconstruct_cell_velocity(f["d_normalVelocity"], f["angleEdge"], f["edgesOnCell"], f["nEdgesOnCell"])
        panel!(fig, 3, 1, xc, yc, sqrt.(u.^2 .+ v.^2), "normalVelocity |u|",        "speed [m/s]")
        panel!(fig, 3, 2, xc, yc, du,                  "∂J/∂normalVelocity (zonal)", "adjoint u"; symmetric = true)
    else
        @warn "no initial_state.nc mesh geometry — plotting raw per-edge normalVelocity (will look noisy)"
        panel!(fig, 3, 1, xe, ye, f["normalVelocity"],   "normalVelocity",     "normalVelocity")
        panel!(fig, 3, 2, xe, ye, f["d_normalVelocity"], "∂J/∂normalVelocity", "adjoint normalVelocity"; symmetric = true)
    end

    Label(fig[0, 1:end], "$(casename(case))  —  primal fields (left) & adjoints (right),  Res: $(res)", fontsize = 24, font = :bold)

    return fig
end

# %% Driver
"""
    run_case(case) -> Figure

Read `output_enzyme.nc` for `case`, print gradient magnitudes, plot the
primal/adjoint fields, and save the figure alongside the data.
"""
function run_case(case::ExampleCase)
    f = read_fields(output_path(case))
    # Mesh geometry for the edge→cell velocity reconstruction lives in initial_state.nc
    # (the enzyme output does not populate angleEdge/connectivity). Best-effort: if it is
    # missing, plot_fields falls back to the raw edge scatter.
    read_mesh_geometry!(f, joinpath(casedir(case), "initial_state.nc"))

    @printf("[%s %s] max |∂J/∂ssh| = %.3e  |∂J/∂layerThickness| = %.3e  |∂J/∂normalVelocity| = %.3e\n",
            casename(case), resolution(case),
            maximum(abs, f["d_ssh"]), maximum(abs, f["d_layerThickness"]),
            maximum(abs, f["d_normalVelocity"]))

    fig = plot_fields(f, case)
    save(joinpath(casedir(case), "adjoint_$(resolution(case)).png"), fig; px_per_unit = 2)
    return fig
end

# %% Main — plot both cases
for case in (InertialGravityWave(), BarotropicGyre())
    set_theme!(fontsize = 18)
    fig = run_case(case)
    
    display(fig)
end
