# %% Dependencies
using NCDatasets
using CairoMakie
using Statistics
using LinearAlgebra
using SparseArrays

# %% Physics parameters (must match config.cfg / setup.py)
const F_0   = 1.0e-3     # Coriolis parameter at southern boundary [s⁻¹]
const BETA  = 1.0e-10    # meridional gradient of Coriolis [s⁻¹ m⁻¹]
const NU_2  = 400.0      # horizontal (Laplacian) viscosity [m² s⁻¹]
const BC    = "no-slip"  # boundary condition: "no-slip" or "free-slip"

# %% Exact (Munk) streamfunction
"""
    exact_streamfunction(x, y; bc) -> Vector{Float64}

Exact barotropic streamfunction (shape) for the linearized Munk experiment,
ported from `Analysis.exact_solution` in the Polaris barotropic_gyre task.
`x`, `y` are vertex coordinates already shifted to the domain origin [m].
"""
function exact_streamfunction(x::Vector{Float64}, y::Vector{Float64};
                              bc::String=BC, beta::Float64=BETA,
                              nu::Float64=NU_2)
    L_x = maximum(x) - minimum(x)
    L_y = maximum(y) - minimum(y)

    delta_m = (nu / (beta * L_y^3))^(1.0 / 3.0)
    gamma   = (sqrt(3.0) .* x) ./ (2.0 * delta_m * L_x)

    if bc == "no-slip"
        psi = π .* sin.(π .* y ./ L_y) .* (
            1.0 .- (x ./ L_x)
            .- exp.(-x ./ (2.0 * delta_m * L_x)) .*
               (cos.(gamma) .+ ((1.0 - 2.0 * delta_m) / sqrt(3.0)) .* sin.(gamma))
            .+ delta_m .* exp.(((x ./ L_x) .- 1.0) ./ delta_m)
        )
    elseif bc == "free-slip"
        psi = π .* sin.(π .* (y ./ L_y)) .* (
            (1.0 .- (x ./ L_x) .- delta_m)
            .+ exp.((-(x ./ L_x)) ./ (2.0 * delta_m)) .*
               ((-2.0 / 3.0) .* (1.0 - delta_m) .* cos.(gamma .- (π / 6.0))
                .+ (2.0 / sqrt(3.0)) .* sin.(gamma))
            .+ delta_m .* exp.((((x ./ L_x) .- 1.0) ./ delta_m))
        )
    else
        error("unknown boundary condition: $bc")
    end
    return psi
end

# %% Numerical barotropic streamfunction
"""
    barotropic_streamfunction(mesh_ds, ssh, nv_surface) -> Vector{Float64}

Compute the barotropic streamfunction at vertices from the surface normal
velocity.  The streamfunction difference across each edge equals the
depth-integrated volume transport through that edge:

    ψ(v₂) − ψ(v₁) = h_edge · u_edge · dvEdge

This linear (incidence) system is rank-deficient by one constant, so vertex 1
is pinned to zero and the system is solved in a least-squares sense.

Single-layer model: the depth integral is just `h_edge · u_edge`, with
`h_edge` the mean layer thickness of the two adjacent cells.
"""
function barotropic_streamfunction(mesh_ds::NCDataset,
                                   layer_thickness::Vector{Float64},
                                   nv_surface::Vector{Float64})
    cellsOnEdge    = Array(mesh_ds["cellsOnEdge"][:, :])     # (2, nEdges)
    verticesOnEdge = Array(mesh_ds["verticesOnEdge"][:, :])  # (2, nEdges)
    dvEdge         = Array(mesh_ds["dvEdge"][:])
    nVertices      = mesh_ds.dim["nVertices"]
    nEdges         = mesh_ds.dim["nEdges"]

    rows = Int[]; cols = Int[]; vals = Float64[]
    b    = Float64[]

    boundary_vertices = Set{Int}()

    for e in 1:nEdges
        v1 = verticesOnEdge[1, e]
        v2 = verticesOnEdge[2, e]
        c1 = cellsOnEdge[1, e]
        c2 = cellsOnEdge[2, e]
        # A wall edge has a missing (0) cell on one side: no normal flow through
        # it, so its vertices lie on the closed boundary streamline.  Record them
        # (skipping any degenerate 0-vertex) and add no transport equation.
        if v1 == 0 || v2 == 0 || c1 == 0 || c2 == 0
            v1 != 0 && (c1 == 0 || c2 == 0) && push!(boundary_vertices, v1)
            v2 != 0 && (c1 == 0 || c2 == 0) && push!(boundary_vertices, v2)
            continue
        end

        h_edge    = 0.5 * (layer_thickness[c1] + layer_thickness[c2])
        transport = h_edge * nv_surface[e] * dvEdge[e]

        # MPAS orientation: normal n̂ (cell1→cell2) and tangent t̂ (vertex1→vertex2)
        # satisfy n̂ × t̂ = +k̂.  With the transport convention (U,V) = k̂ × ∇ψ used
        # by the exact Munk solution, the flux in the +n̂ direction equals
        # ψ(v1) − ψ(v2), hence ψ(v2) − ψ(v1) = −transport.
        push!(rows, length(b) + 1); push!(cols, v2); push!(vals,  1.0)
        push!(rows, length(b) + 1); push!(cols, v1); push!(vals, -1.0)
        push!(b, -transport)
    end

    # The solid walls form a single closed streamline (no normal flow), so ψ is
    # constant there; pin every boundary vertex to 0.  This both removes the
    # constant null space and anchors the wall isoline, which is where the error
    # otherwise concentrates.  Fall back to pinning vertex 1 if no wall was found.
    if isempty(boundary_vertices)
        push!(boundary_vertices, 1)
    end
    for v in boundary_vertices
        push!(rows, length(b) + 1); push!(cols, v); push!(vals, 1.0)
        push!(b, 0.0)
    end

    A = sparse(rows, cols, vals, length(b), nVertices)
    psi = A \ b
    return psi
end

# %% Read MOKA output and compare
function read_and_compare(out_fp::String, mesh_fp::String; frame::Int=0)
    mesh_ds = NCDataset(mesh_fp)
    num_ds  = NCDataset(out_fp)

    # shift vertex coordinates to the domain origin (matches Polaris)
    xVertex = Array(mesh_ds["xVertex"][:]); xVertex .-= minimum(Array(mesh_ds["xEdge"][:]))
    yVertex = Array(mesh_ds["yVertex"][:]); yVertex .-= minimum(Array(mesh_ds["yEdge"][:]))

    # surface fields at the selected output time (default: last frame)
    t = frame == 0 ? size(num_ds["layerThickness"], 3) : frame
    layer_thickness = Array(num_ds["layerThickness"][:, 1, t])  # (nCells,) surface
    nv_surface      = Array(num_ds["normalVelocity"][:, 1, t])  # (nEdges,) surface

    psi_num = barotropic_streamfunction(mesh_ds, layer_thickness, nv_surface)
    psi_ext = exact_streamfunction(xVertex, yVertex)

    # both are defined up to an arbitrary amplitude/offset; normalise to
    # zero-mean, unit-max so the comparison is on shape (Munk gyre pattern)
    normalize_field(v) = (v .- mean(v)) ./ maximum(abs.(v .- mean(v)))
    psi_num_n = normalize_field(psi_num)
    psi_ext_n = normalize_field(psi_ext)

    # area-weighted, exact-normalised L2 error (mirrors Polaris compute_error)
    area = Array(mesh_ds["areaTriangle"][:])
    diff = psi_ext_n .- psi_num_n
    l2_err = norm(diff .* area) / norm(psi_ext_n .* area)

    close(num_ds)

    results = Dict(
        "xVertex" => xVertex,
        "yVertex" => yVertex,
        "psi_num" => psi_num_n,
        "psi_ext" => psi_ext_n,
        "psi_err" => diff,
        "l2_err"  => l2_err,
    )
    return results, mesh_ds
end

# %% Three-panel comparison plot (numerical / analytical / error)
function plot_comparison(results::Dict, res_num::String="")
    x = results["xVertex"] ./ 1e3
    y = results["yVertex"] ./ 1e3

    lim = maximum(abs.(results["psi_ext"]))
    fig = Figure(size = (1200, 420))

    ax1 = Axis(fig[1, 1], title="Numerical ψ, Res: $(res_num)",
               xlabel="x (km)", ylabel="y (km)", aspect=1)
    s1 = scatter!(ax1, x, y, color=results["psi_num"], colormap=:balance,
                  colorrange=(-lim, lim), markersize=5)
    Colorbar(fig[1, 2], s1)

    ax2 = Axis(fig[1, 3], title="Analytical ψ (Munk, $(BC))",
               xlabel="x (km)", aspect=1)
    s2 = scatter!(ax2, x, y, color=results["psi_ext"], colormap=:balance,
                  colorrange=(-lim, lim), markersize=5)
    Colorbar(fig[1, 4], s2)

    elim = maximum(abs.(results["psi_err"]))
    ax3 = Axis(fig[1, 5], title="Error (Numerical − Analytical)",
               xlabel="x (km)", aspect=1)
    s3 = scatter!(ax3, x, y, color=results["psi_err"], colormap=:coolwarm,
                  colorrange=(-elim, elim), markersize=5)
    Colorbar(fig[1, 6], s3)

    return fig
end

# %% Main
# Guarded so this file can be `include`d as a library (e.g. by convergence.jl)
# without running the single-resolution comparison / plotting side effects.
if abspath(PROGRAM_FILE) == @__FILE__
    res     = "10km"
    dir_    = joinpath(@__DIR__, res)
    mesh_fp = joinpath(dir_, "initial_state.nc")
    out_fp  = joinpath(dir_, "output.nc")

    results, mesh_ds = read_and_compare(out_fp, mesh_fp)
    close(mesh_ds)

    println("L2 error norm for $(BC) barotropic streamfunction: ",
            round(results["l2_err"]; digits=4))

    fig = plot_comparison(results, res)
    save(joinpath(@__DIR__, "comparison.png"), fig; px_per_unit = 3)
    display(fig)
end
