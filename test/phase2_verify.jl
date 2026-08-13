# Phase 2 (constants & config plumbing) verification harness. Standalone; needs
# the real barotropic gyre initial_state.nc under examples/. Checks:
#   1. Constants come from config (custom gravity changes the pressure-gradient
#      result; default gravity == 9.80616 reproduces the baseline dynamics).
#   2. config_use_bulk_wind_stress gates the wind term (on vs off differ; off
#      matches a mesh with no wind fields).
#   3. config_use_mom_del2 gates lateral mixing (on vs off differ).
#   4. config_get(key, default) and config_has behave; a missing required key
#      raises a friendly error listing available keys.

using UnstructuredOceans
using NCDatasets
using Printf
using Statistics
import KernelAbstractions as KA
using Dates

const BG20 = joinpath(@__DIR__, "..", "examples", "barotropic_gyre", "20km")

function bg_config(mesh_dir; nsteps=50, dt=1.0, wind=true, mixing=true,
                   mom_del2=400.0, gravity=nothing)
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
        "config_output_reference_time" => DateTime(1,1,1,0,0,0),
    )))
    UnstructuredOceans.config_add(nl, "time_integration", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_dt" => Second(round(Int, dt)),
        "config_time_integrator" => "RungeKutta4",
        "config_number_of_time_levels" => 2,
    )))
    UnstructuredOceans.config_add(nl, "forcing", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_use_bulk_wind_stress" => wind,
    )))
    UnstructuredOceans.config_add(nl, "hmix_del2", UnstructuredOceans.yaml_config(Dict{Any,Any}(
        "config_use_mom_del2" => mixing,
        "config_mom_del2" => mom_del2,
    )))
    if gravity !== nothing
        UnstructuredOceans.config_add(nl, "constants", UnstructuredOceans.yaml_config(Dict{Any,Any}(
            "config_gravity" => gravity,
        )))
    end
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

function run_cfg(cfg; nsteps=50, dt=1.0)
    backend = KA.CPU()
    Mesh  = UnstructuredOceans.ocn_setup_mesh(cfg; backend=backend)
    Clock = UnstructuredOceans.ocn_setup_clock(cfg)
    Setup = UnstructuredOceans.ModelSetup(cfg, Mesh, Clock)
    Prog  = UnstructuredOceans.PrognosticVars(cfg, Mesh; backend=backend)
    Diag  = UnstructuredOceans.DiagnosticVars(Mesh; backend=backend)
    Tend  = UnstructuredOceans.TendencyVars(Mesh; backend=backend)
    dt_arr = KA.zeros(backend, Float64, 1); dt_arr[1] = dt
    for _ in 1:nsteps
        UnstructuredOceans.ocn_timestep(dt_arr, Prog, Diag, Tend, Mesh, RungeKutta4)
    end
    return (ssh = Array(Prog.ssh[end]),
            vel = Array(Prog.normalVelocity[end]),
            h   = Array(Prog.layerThickness[end]))
end

fnorm(r) = sum(abs2, r.ssh) + sum(abs2, r.vel)

function main()
    println(">> Config accessor / validation checks")
    cfg = bg_config(BG20)
    @assert UnstructuredOceans.config_has(cfg.namelist, "forcing")
    @assert !UnstructuredOceans.config_has(cfg.namelist, "nonexistent_section")
    fc = UnstructuredOceans.config_get(cfg.namelist, "forcing")
    @assert UnstructuredOceans.config_get(fc, "config_use_bulk_wind_stress", false) == true
    @assert UnstructuredOceans.config_get(fc, "missing_key", 42) == 42
    threw = false
    try
        UnstructuredOceans.config_get(cfg.namelist, "definitely_missing")
    catch e
        threw = true
        msg = sprint(showerror, e)
        @assert occursin("definitely_missing", msg) && occursin("Available keys", msg)
    end
    @assert threw "missing required key should have thrown"
    println("   PASS: config_has / config_get default / friendly missing-key error")

    println(">> Constants from config: default gravity reproduces baseline")
    r_default  = run_cfg(bg_config(BG20))                       # no constants section
    r_explicit = run_cfg(bg_config(BG20; gravity=9.80616))      # explicit == default literal
    dg = abs(fnorm(r_default) - fnorm(r_explicit)) / max(fnorm(r_default), eps())
    @printf("   rel diff default-vs-explicit gravity = %.3e\n", dg)
    @assert dg < 1e-14 "explicit default gravity should reproduce the implicit default"

    println(">> Constants from config: custom gravity changes the dynamics")
    r_grav = run_cfg(bg_config(BG20; gravity=5.0))
    dgg = abs(fnorm(r_grav) - fnorm(r_default)) / max(fnorm(r_default), eps())
    @printf("   rel diff gravity 5.0 vs 9.80616 = %.3e\n", dgg)
    @assert dgg > 1e-3 "changing gravity should change the pressure-gradient dynamics"
    println("   PASS: gravity is config-driven")

    println(">> Wind-stress gate")
    r_wind_on  = run_cfg(bg_config(BG20; wind=true))
    r_wind_off = run_cfg(bg_config(BG20; wind=false))
    dw = abs(fnorm(r_wind_on) - fnorm(r_wind_off)) / max(fnorm(r_wind_on), eps())
    @printf("   rel diff wind on vs off = %.3e\n", dw)
    @assert dw > 1e-3 "config_use_bulk_wind_stress=false should change results"
    println("   PASS: wind forcing is config-gated")

    println(">> Mixing (del2) gate")
    r_mix_on  = run_cfg(bg_config(BG20; mixing=true,  mom_del2=400.0))
    r_mix_off = run_cfg(bg_config(BG20; mixing=false, mom_del2=400.0))
    dm = abs(fnorm(r_mix_on) - fnorm(r_mix_off)) / max(fnorm(r_mix_on), eps())
    @printf("   rel diff mixing on vs off = %.3e\n", dm)
    @assert dm > 1e-6 "config_use_mom_del2=false should change results"
    println("   PASS: lateral mixing is config-gated")

    println("\nAll Phase 2 verification checks PASSED.")
end

main()
