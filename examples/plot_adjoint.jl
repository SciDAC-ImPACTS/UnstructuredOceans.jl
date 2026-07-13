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

    fields = Dict(
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

    fig = Figure(size = (1500, 900))
    Label(fig[0, 1:6], "$(casename(case))  —  primal fields (top) & adjoints (bottom),  Res: $(res)",
          fontsize = 20, font = :bold)

    # Row 1 — primal
    panel!(fig, 1, 1, xc, yc, f["ssh"],            "ssh",            "ssh")
    panel!(fig, 1, 2, xc, yc, f["layerThickness"], "layerThickness", "layerThickness")
    panel!(fig, 1, 3, xe, ye, f["normalVelocity"], "normalVelocity", "normalVelocity")

    # Row 2 — adjoint / gradient
    panel!(fig, 2, 1, xc, yc, f["d_ssh"],            "∂J/∂ssh",            "adjoint ssh";            symmetric = true)
    panel!(fig, 2, 2, xc, yc, f["d_layerThickness"], "∂J/∂layerThickness", "adjoint layerThickness"; symmetric = true)
    panel!(fig, 2, 3, xe, ye, f["d_normalVelocity"], "∂J/∂normalVelocity", "adjoint normalVelocity"; symmetric = true)

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
    fig = run_case(case)
    display(fig)
end
