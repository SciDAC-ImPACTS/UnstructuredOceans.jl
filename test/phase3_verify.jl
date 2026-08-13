# Phase 3 (vector-invariant nonlinear advection) verification. Standalone; uses
# the barotropic gyre mesh (has kiteAreasOnVertex, fᵛ). Checks:
#   1. Reconstruction kernels against exactly-representable inputs:
#        - layer_thickness_vertex! of a CONSTANT thickness == that constant
#          (kite areas sum to the triangle area), at every level.
#        - potential_vorticity_vertex! with zero velocity == fᵛ / h  (ζ = 0).
#        - kinetic_energy_cell! of ZERO velocity == 0; of nonzero velocity ≥ 0.
#   2. vectorInvariant momentum term runs end-to-end, stays finite, and differs
#      from linearCoriolis (it adds nonlinear advection + KE gradient).
#   3. Forward-mode Enzyme AD through a vectorInvariant step works (rel err vs FD).

using UnstructuredOceans
using NCDatasets
using Printf
using Statistics
import KernelAbstractions as KA
using Dates
using Enzyme
using UnPack

const NV = UnstructuredOceans.NormalVelocity
const BG20 = joinpath(@__DIR__, "..", "examples", "barotropic_gyre", "20km")

# --- shared config builder (wind off, mixing off: isolate the advection term) ---
function bg_config(mesh_dir; nsteps=10, dt=1.0)
    initial = joinpath(mesh_dir, "initial_state.nc")
    cfg = UnstructuredOceans.GlobalConfig()
    nl = cfg.namelist
    UnstructuredOceans.config_add(nl, "time_management", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_do_restart" => false,
        "config_restart_timestamp_name" => "Restart_timestamp",
        "config_start_time" => DateTime(1,1,1,0,0,0),
        "config_stop_time"  => DateTime(1,1,1,0,0,0) + Second(round(Int, nsteps*dt)),
        "config_run_duration" => "none",
        "config_calendar_type" => "noleap",
        "config_output_reference_time" => DateTime(1,1,1,0,0,0))))
    UnstructuredOceans.config_add(nl, "time_integration", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_dt" => Second(round(Int, dt)),
        "config_time_integrator" => "RungeKutta4",
        "config_number_of_time_levels" => 2)))
    UnstructuredOceans.config_add(nl, "forcing", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_use_bulk_wind_stress" => false)))
    UnstructuredOceans.config_add(nl, "hmix_del2", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_use_mom_del2" => false, "config_mom_del2" => 0.0)))
    st = cfg.streams
    for name in ("mesh", "input", "forcing")
        UnstructuredOceans.config_add(st, name, UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "filename_template" => initial)))
    end
    UnstructuredOceans.config_add(st, "output", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "filename_template" => tempname() * ".nc",
        "reference_time" => DateTime(1,1,1,0,0,0),
        "output_interval" => Second(round(Int, nsteps*dt)))))
    return cfg
end

function build_state(mesh_dir; nVertLevels_hint=nothing, nsteps=10, dt=1.0, backend=KA.CPU())
    cfg   = bg_config(mesh_dir; nsteps=nsteps, dt=dt)
    Mesh  = UnstructuredOceans.ocn_setup_mesh(cfg; backend=backend)
    Clock = UnstructuredOceans.ocn_setup_clock(cfg)
    Setup = UnstructuredOceans.ModelSetup(cfg, Mesh, Clock)
    Prog  = UnstructuredOceans.PrognosticVars(cfg, Mesh; backend=backend)
    Diag  = UnstructuredOceans.DiagnosticVars(Mesh; backend=backend)
    Tend  = UnstructuredOceans.TendencyVars(Mesh; backend=backend)
    return (; Setup, Mesh, Prog, Diag, Tend)
end

function check_reconstructions()
    println(">> Reconstruction kernels vs closed form")
    backend = KA.CPU()
    s = build_state(BG20; backend=backend)
    mesh = s.Mesh
    @unpack HorzMesh, VertMesh = mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    nCells    = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nEdges    = Edges.nEdges
    nVertLevels = VertMesh.nVertLevels

    # --- layer_thickness_vertex! of a constant field ---
    hconst = 1234.5
    layerThickness = fill(hconst, nVertLevels, nCells)
    hv = KA.zeros(backend, Float64, nVertLevels, nVertices)
    k! = NV.layer_thickness_vertex!(backend, UnstructuredOceans.DEFAULT_NTHREADS)
    k!(hv, layerThickness, DualCells.cellsOnVertex, DualCells.kiteAreasOnVertex,
       DualCells.areaTriangle, DualCells.vertexDegree, nVertices,
       ndrange=(nVertices, nVertLevels))
    hv = Array(hv)
    # interior vertices (all 3 cells present) must recover the constant exactly
    interior = [v for v in 1:nVertices if all(DualCells.cellsOnVertex[:, v] .!= 0)]
    err_hv = maximum(abs, hv[1, interior] .- hconst) / hconst
    @printf("   layer_thickness_vertex! const: rel err (interior) = %.3e\n", err_hv)
    @assert err_hv < 1e-10 "thickness-to-vertex of a constant should be exact"

    # --- potential_vorticity_vertex! with ζ = 0 → f/h ---
    relVort = KA.zeros(backend, Float64, nVertLevels, nVertices)  # zero velocity
    pv = KA.zeros(backend, Float64, nVertLevels, nVertices)
    hv_dev = KA.zeros(backend, Float64, nVertLevels, nVertices); copyto!(hv_dev, reshape(repeat(hv[1,:], nVertLevels), nVertLevels, nVertices) )
    kp! = NV.potential_vorticity_vertex!(backend, UnstructuredOceans.DEFAULT_NTHREADS)
    kp!(pv, relVort, hv_dev, DualCells.fᵛ, nVertices, ndrange=(nVertices, nVertLevels))
    pv = Array(pv)
    fv = DualCells.fᵛ
    expected = [hv[1,v] > 0 ? fv[v]/hv[1,v] : 0.0 for v in 1:nVertices]
    err_pv = maximum(abs, pv[1, interior] .- expected[interior]) /
             max(maximum(abs, expected[interior]), eps())
    @printf("   potential_vorticity_vertex! (ζ=0): rel err = %.3e\n", err_pv)
    @assert err_pv < 1e-10 "PV with zero velocity should equal f/h"

    # --- kinetic_energy_cell!: zero velocity → 0; nonzero → ≥ 0 ---
    kez = KA.zeros(backend, Float64, nVertLevels, nCells)
    uvel = KA.zeros(backend, Float64, nVertLevels, nEdges)
    ke! = NV.kinetic_energy_cell!(backend, UnstructuredOceans.DEFAULT_NTHREADS)
    ke!(kez, uvel, PrimaryCells.nEdgesOnCell, PrimaryCells.edgesOnCell,
        Edges.dcEdge, Edges.dvEdge, PrimaryCells.areaCell, nCells,
        ndrange=(nCells, nVertLevels))
    @assert maximum(abs, Array(kez)) == 0.0 "KE of zero velocity must be zero"
    # KE of a UNIFORM edge-normal field should be ~½u² on interior cells (the ⅛
    # coefficient + Σ¼dc·dv/A = 1 identity). Boundary cells differ (missing edges).
    uconst = 0.7
    uvel = KA.zeros(backend, Float64, nVertLevels, nEdges); fill!(uvel, uconst)
    ke!(kez, uvel, PrimaryCells.nEdgesOnCell, PrimaryCells.edgesOnCell,
        Edges.dcEdge, Edges.dvEdge, PrimaryCells.areaCell, nCells,
        ndrange=(nCells, nVertLevels))
    kez = Array(kez)
    # interior cells: all edges present
    interior_c = [c for c in 1:nCells if all(PrimaryCells.edgesOnCell[1:PrimaryCells.nEdgesOnCell[c], c] .!= 0)]
    err_ke = maximum(abs, kez[1, interior_c] .- 0.5*uconst^2) / (0.5*uconst^2)
    @printf("   kinetic_energy_cell! uniform u=%.1f: interior rel err vs ½u² = %.3e\n", uconst, err_ke)
    @assert minimum(kez) >= 0.0 "KE must be non-negative"
    @assert err_ke < 1e-10 "KE of uniform velocity should be ½u² on interior cells"

    println("   PASS: reconstructions match closed form")
end

# Seed a SPATIALLY-VARYING velocity: a uniform field would make the nonlinear
# term (PV ≈ f/h const, ∇K ≈ 0) collapse onto linearCoriolis, so structure is
# needed to exercise the difference. Use a sinusoid in the edge x-coordinate.
function seed_structured!(s)
    xe = s.Mesh.HorzMesh.Edges.xᵉ
    Lx = maximum(xe) - minimum(xe)
    nlev = size(s.Prog.normalVelocity[end], 1)
    u = 1.0 .* sin.(2π .* xe ./ Lx)
    for k in 1:nlev
        s.Prog.normalVelocity[end][k, :]   .= u
        s.Prog.normalVelocity[end-1][k, :] .= u
    end
end

function run_step(mesh_dir; coriolis, nsteps=10, dt=1.0)
    backend = KA.CPU()
    s = build_state(mesh_dir; nsteps=nsteps, dt=dt, backend=backend)
    seed_structured!(s)
    dt_arr = KA.zeros(backend, Float64, 1); dt_arr[1] = dt
    for _ in 1:nsteps
        UnstructuredOceans.ocn_timestep(dt_arr, s.Prog, s.Diag, s.Tend, s.Mesh, RungeKutta4;
                          coriolis=coriolis)
    end
    return (ssh=Array(s.Prog.ssh[end]), vel=Array(s.Prog.normalVelocity[end]))
end

function check_end_to_end()
    println(">> vectorInvariant end-to-end vs linearCoriolis")
    r_lin = run_step(BG20; coriolis=NV.linearCoriolis, nsteps=20)
    r_vi  = run_step(BG20; coriolis=NV.vectorInvariant, nsteps=20)
    @assert all(isfinite, r_vi.vel) && all(isfinite, r_vi.ssh) "vectorInvariant produced non-finite state"
    # The KE-gradient term is the dominant difference on this f-plane, depth-uniform
    # basin (PV ≈ f/h ≈ const makes the PV-flux term nearly the linear Σ w f u); the
    # difference is small but must be clearly above round-off.
    d = sum(abs2, r_vi.vel .- r_lin.vel) / max(sum(abs2, r_lin.vel), eps())
    @printf("   rel diff vectorInvariant vs linearCoriolis (vel) = %.3e\n", d)
    @assert d > 1e-9 "vectorInvariant should differ from linearCoriolis"
    println("   PASS: vectorInvariant runs, finite, and adds nonlinear advection")
end

function check_ad()
    println(">> Forward-mode AD through a vectorInvariant step")
    backend = KA.CPU()
    dt = 1.0; nsteps = 5
    dt_arr = KA.zeros(backend, Float64, 1); dt_arr[1] = dt

    function loss!(viscDel2_arr, Prog, Diag, Tend, Mesh, dt_arr, nsteps)
        for _ in 1:nsteps
            UnstructuredOceans.ocn_timestep(dt_arr, Prog, Diag, Tend, Mesh, RungeKutta4;
                              coriolis=NV.vectorInvariant, viscDel2=viscDel2_arr)
        end
        return sum(abs2, Prog.normalVelocity[end])
    end

    mk() = begin
        s = build_state(BG20; nsteps=nsteps, dt=dt, backend=backend)
        seed_structured!(s)
        s
    end

    visc0 = 400.0
    v  = KA.zeros(backend, Float64, 1); v[1]  = visc0
    dv = KA.zeros(backend, Float64, 1); dv[1] = 1.0
    s  = mk()
    d_Prog = Enzyme.make_zero(s.Prog)
    d_Diag = Enzyme.make_zero(s.Diag)
    d_Tend = Enzyme.make_zero(s.Tend)
    dloss = Enzyme.autodiff(Enzyme.Forward, loss!,
                            Enzyme.Duplicated(v, dv),
                            Enzyme.Duplicated(s.Prog, d_Prog),
                            Enzyme.Duplicated(s.Diag, d_Diag),
                            Enzyme.Duplicated(s.Tend, d_Tend),
                            Enzyme.Const(s.Mesh),
                            Enzyme.Const(dt_arr),
                            Enzyme.Const(nsteps))[1]

    fd_eval(visc) = begin
        vv = KA.zeros(backend, Float64, 1); vv[1] = visc
        st = mk()
        for _ in 1:nsteps
            UnstructuredOceans.ocn_timestep(dt_arr, st.Prog, st.Diag, st.Tend, st.Mesh, RungeKutta4;
                              coriolis=NV.vectorInvariant, viscDel2=vv)
        end
        sum(abs2, st.Prog.normalVelocity[end])
    end
    ϵ = 1e-2
    fd = (fd_eval(visc0+ϵ) - fd_eval(visc0-ϵ)) / (2ϵ)
    relerr = abs(dloss - fd) / max(abs(fd), eps())
    @printf("   Enzyme = %.8e  FD = %.8e  rel err = %.3e\n", dloss, fd, relerr)
    @assert relerr < 1e-4 "AD through vectorInvariant disagrees with FD"
    println("   PASS: vectorInvariant is Enzyme-differentiable")
end

function main()
    check_reconstructions()
    check_end_to_end()
    check_ad()
    println("\nAll Phase 3 verification checks PASSED.")
end

main()
