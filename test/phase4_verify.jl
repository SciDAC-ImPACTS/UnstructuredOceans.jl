# Phase 4 (forcing framework) verification. Standalone; uses the barotropic gyre
# mesh (carries windStressZonal/Meridional). Checks:
#   1. ForcingVars.windStressEdge matches the reference edge-normal projection of
#      the wind stress / rho (i.e. the relocation off Edges is bit-faithful).
#   2. use_wind_stress=false yields all-zero forcing.
#   3. Empty ForcingVars(nEdges) is zero; Mesh backward-compat constructors carry
#      a zeroed ForcingVars.
#   4. WindForcing dispatch is reachable and the forcing tuple is configurable
#      (empty tuple => no wind tendency, matching wind-off).

using UnstructuredOceans
using NCDatasets
using Printf
import KernelAbstractions as KA
using Dates
using UnPack

const NV = UnstructuredOceans.NormalVelocity
const BG20 = joinpath(@__DIR__, "..", "examples", "barotropic_gyre", "20km")
const INIT = joinpath(BG20, "initial_state.nc")

function reference_wind_projection(h_mesh; rho=1000.0)
    edges = h_mesh.Edges
    nEdges = edges.nEdges
    angleEdge   = Array(edges.angleEdge)
    cellsOnEdge = Array(edges.cellsOnEdge)
    ds = NCDataset(INIT, "r")
    τz_c = Float64.(ds["windStressZonal"][:, 1])
    τm_c = Float64.(ds["windStressMeridional"][:, 1])
    close(ds)
    w = zeros(Float64, nEdges)
    for iEdge in 1:nEdges
        c1 = cellsOnEdge[1, iEdge]; c2 = cellsOnEdge[2, iEdge]
        τz_e = (c1 == 0 ? τz_c[c2] : (c2 == 0 ? τz_c[c1] : 0.5*(τz_c[c1]+τz_c[c2])))
        τm_e = (c1 == 0 ? τm_c[c2] : (c2 == 0 ? τm_c[c1] : 0.5*(τm_c[c1]+τm_c[c2])))
        w[iEdge] = (τz_e*cos(angleEdge[iEdge]) + τm_e*sin(angleEdge[iEdge])) / rho
    end
    return w
end

function main()
    backend = KA.CPU()

    println(">> ForcingVars projection matches reference")
    h = read_horz_mesh(INIT; backend=backend)
    f = UnstructuredOceans.MPASMesh.build_forcing(h, INIT; rho=1000.0, use_wind_stress=true, backend=backend)
    wref = reference_wind_projection(h; rho=1000.0)
    err = maximum(abs, Array(f.windStressEdge) .- wref) / max(maximum(abs, wref), eps())
    @printf("   rel err vs reference projection = %.3e  (Σ|w| = %.4f)\n", err, sum(abs, wref))
    @assert err == 0.0 "ForcingVars projection should be bit-identical to the reference"
    @assert sum(abs, wref) > 0 "reference wind should be nonzero (sanity)"

    println(">> use_wind_stress=false yields zero forcing")
    f0 = UnstructuredOceans.MPASMesh.build_forcing(h, INIT; use_wind_stress=false, backend=backend)
    @assert maximum(abs, Array(f0.windStressEdge)) == 0.0 "wind-off forcing must be zero"

    println(">> empty ForcingVars + Mesh backward-compat constructors are zeroed")
    fe = ForcingVars(h.Edges.nEdges)
    @assert maximum(abs, Array(fe.windStressEdge)) == 0.0
    vm = VerticalMesh(h; nVertLevels=1, backend=backend)
    mesh2 = Mesh(h, vm)             # 2-arg: default constants + zeroed forcing
    @assert maximum(abs, Array(mesh2.Forcing.windStressEdge)) == 0.0
    @assert size(mesh2.Forcing.windStressEdge) == (h.Edges.nEdges,)
    println("   PASS: forcing construction paths correct")

    println(">> WindForcing dispatch + configurable forcing tuple")
    # Build a full state with wind on; compare the default forcing tuple
    # (WindForcing,) against an empty tuple () — the latter must equal a wind-off run.
    function run(mesh_dir; forcings, wind_in_mesh=true, nsteps=20, dt=1.0)
        cfg = UnstructuredOceans.GlobalConfig(); nl = cfg.namelist
        UnstructuredOceans.config_add(nl, "time_management", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "config_do_restart"=>false, "config_restart_timestamp_name"=>"R",
            "config_start_time"=>DateTime(1,1,1), "config_run_duration"=>"none",
            "config_stop_time"=>DateTime(1,1,1)+Second(nsteps),
            "config_calendar_type"=>"noleap", "config_output_reference_time"=>DateTime(1,1,1))))
        UnstructuredOceans.config_add(nl, "time_integration", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "config_dt"=>Second(round(Int,dt)), "config_time_integrator"=>"RungeKutta4",
            "config_number_of_time_levels"=>2)))
        UnstructuredOceans.config_add(nl, "forcing", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "config_use_bulk_wind_stress"=>wind_in_mesh)))
        UnstructuredOceans.config_add(nl, "hmix_del2", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "config_use_mom_del2"=>false, "config_mom_del2"=>0.0)))
        st = cfg.streams
        for name in ("mesh","input","forcing")
            UnstructuredOceans.config_add(st, name, UnstructuredOceans.yaml_config(Dict{Any,Any}("filename_template"=>INIT)))
        end
        UnstructuredOceans.config_add(st, "output", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "filename_template"=>tempname()*".nc", "reference_time"=>DateTime(1,1,1),
            "output_interval"=>Second(nsteps))))
        Mesh  = UnstructuredOceans.ocn_setup_mesh(cfg; backend=backend)
        Clock = UnstructuredOceans.ocn_setup_clock(cfg)
        Setup = UnstructuredOceans.ModelSetup(cfg, Mesh, Clock)
        Prog  = UnstructuredOceans.PrognosticVars(cfg, Mesh; backend=backend)
        Diag  = UnstructuredOceans.DiagnosticVars(Mesh; backend=backend)
        Tend  = UnstructuredOceans.TendencyVars(Mesh; backend=backend)
        dt_arr = KA.zeros(backend, Float64, 1); dt_arr[1] = dt
        for _ in 1:nsteps
            UnstructuredOceans.ocn_timestep(dt_arr, Prog, Diag, Tend, Mesh, RungeKutta4; forcings=forcings)
        end
        return Array(Prog.normalVelocity[end])
    end

    v_wind    = run(BG20; forcings=(NV.WindForcing,), wind_in_mesh=true)
    v_noforce = run(BG20; forcings=(),                wind_in_mesh=true)
    v_meshoff = run(BG20; forcings=(NV.WindForcing,), wind_in_mesh=false)
    # empty forcing tuple == wind zeroed in mesh (both apply no wind tendency)
    d1 = maximum(abs, v_noforce .- v_meshoff)
    @printf("   max|empty-tuple - mesh-wind-off| = %.3e\n", d1)
    @assert d1 < 1e-14 "empty forcing tuple should match wind disabled in mesh"
    # wind on must differ from no-forcing
    d2 = sum(abs2, v_wind .- v_noforce) / max(sum(abs2, v_wind), eps())
    @printf("   rel diff wind-on vs no-forcing = %.3e\n", d2)
    @assert d2 > 1e-6 "WindForcing must change the tendency"
    println("   PASS: forcing dispatch + tuple configuration work")

    println("\nAll Phase 4 verification checks PASSED.")
end

main()
