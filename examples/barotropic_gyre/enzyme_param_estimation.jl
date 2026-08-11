using Test
using Dates
import KernelAbstractions as KA
using Enzyme
using MOKA
using CUDA
import CUDA: @allowscalar
using Optimization
using OptimizationOptimJL
using NCDatasets
using CairoMakie
using Printf
CUDA.device!(1)

# Parameter estimation for the barotropic gyre: recover the del2 momentum
# viscosity (config_mom_del2) that best reproduces a *reference* final state
# generated with a known "true" viscosity.
#
# This builds directly on enzyme_forward_test.jl. There, one forward-mode AD
# sweep of `ocn_run_loop_fwd!` yields ∂(final state)/∂(viscosity) for every
# state entry, seeding `viscDel2` as a Duplicated argument with tangent 1.
# Here we reuse that machinery as the gradient oracle for an inverse problem:
#
#     minimize   J(ν) = ‖ h(ν) − h_ref ‖² + ‖ u(ν) − u_ref ‖²
#        ν
#
# where h, u are the final layerThickness / normalVelocity fields and h_ref,
# u_ref come from a forward run at the true viscosity ν★. Because ν is a single
# scalar, forward mode is the ideal gradient source: a single sweep gives
# ∂h/∂ν and ∂u/∂ν over the whole grid, and the chain rule collapses the
# objective gradient to
#
#     dJ/dν = Σ 2 (h − h_ref)·(∂h/∂ν) + Σ 2 (u − u_ref)·(∂u/∂ν).
#
# We hand J and dJ/dν to Optimization.jl and let a gradient-based optimizer
# (Optim's LBFGS via a box constraint) drive ν back to ν★.

function _nsteps(Setup)
    clock, simAlarm, _ = ocn_init_alarms(Setup)
    total_ms = Dates.value(Millisecond(simAlarm.ringTime - clock.startTime))
    dt_ms    = Dates.value(Millisecond(clock.timeStep))
    return Int(total_ms ÷ dt_ms)
end

# Everything needed to advance the model that does NOT depend on viscosity, built
# once and reused every optimizer iteration so each evaluation only pays for the
# time integration itself (not re-reading the mesh / initial state from disk).
struct GyreProblem{S,D,T,P,TS,M,I}
    Setup::S
    Diag::D
    Tend::T
    Prog0::P            # pristine initial Prog (deep-copied per run)
    timestep::TS
    Mesh::M
    integrator::I
    nsteps::Int
    backend::Any
end

function GyreProblem(config_fp, backend)
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend=backend)
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
    Mesh = Setup.mesh
    integrator = parse_integrator(
        MOKA.config_get(MOKA.config_get(Setup.config.namelist, "time_integration"),
                       "config_time_integrator"))
    nsteps = _nsteps(Setup)
    return GyreProblem(Setup, Diag, Tend, Prog, timestep, Mesh, integrator, nsteps, backend)
end

# A fresh state bundle for one model run: Prog reset to the initial condition and
# Diag/Tend zeroed, so successive evaluations are independent.
function fresh_state(gp::GyreProblem)
    Prog = deepcopy(gp.Prog0)
    Diag = Enzyme.make_zero(gp.Diag)
    Tend = Enzyme.make_zero(gp.Tend)
    return Prog, Diag, Tend
end

# Advance the model to the final time at viscosity `visc` and return the final
# (layerThickness, normalVelocity) fields as *host* arrays.
function forward_state(gp::GyreProblem, visc::Float64)
    Prog, Diag, Tend = fresh_state(gp)
    viscDel2 = KA.zeros(gp.backend, Float64, (1,))
    @allowscalar viscDel2[1] = visc
    ocn_run_loop_fwd!(viscDel2, gp.timestep, Prog, Diag, Tend,
                      gp.Mesh, gp.integrator, gp.nsteps)
    return Array(Prog.layerThickness[end]), Array(Prog.normalVelocity[end])
end

# The raw residual objective is tiny in absolute units — the state residuals are
# O(1e-5) and the viscosity sensitivities O(1e-10), so J ≈ c·(ν−ν★)² with
# c ~ 1e-10 and dJ/dν lands at ~1e-8 even a few units away from the optimum. A
# gradient-based optimizer with any reasonable g_tol would then stop far short of
# ν★, not because it is stuck but because the gradient criterion trips first.
# Rescaling J (and hence its gradient) by a large constant leaves the minimizer
# unchanged but lifts the gradient to O(1), so default tolerances behave sanely.
# This is the standard remedy for a badly-scaled inverse problem.
const LOSS_SCALE = 1e8

# Objective J(ν) and its gradient dJ/dν in one forward-mode AD sweep.
# `h_ref`, `u_ref` are the reference (target) final fields as host arrays.
function loss_and_grad(gp::GyreProblem, visc::Float64, h_ref, u_ref)
    Prog, Diag, Tend = fresh_state(gp)

    viscDel2 = KA.zeros(gp.backend, Float64, (1,))
    @allowscalar viscDel2[1] = visc
    d_viscDel2 = KA.zeros(gp.backend, Float64, (1,))
    @allowscalar d_viscDel2[1] = 1.0        # seed tangent: differentiate w.r.t. ν

    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)

    autodiff(Forward, ocn_run_loop_fwd!,
             Duplicated(viscDel2, d_viscDel2),
             Const(gp.timestep),
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Const(gp.Mesh),
             Const(gp.integrator),
             Const(gp.nsteps))

    # Final fields and their viscosity sensitivities, brought to the host.
    h   = Array(Prog.layerThickness[end]);  dh = Array(d_Prog.layerThickness[end])
    u   = Array(Prog.normalVelocity[end]);  du = Array(d_Prog.normalVelocity[end])

    rh = h .- h_ref
    ru = u .- u_ref
    J     = LOSS_SCALE * (sum(abs2, rh) + sum(abs2, ru))
    dJ_dν = LOSS_SCALE * 2.0 * (sum(rh .* dh) + sum(ru .* du))
    return J, dJ_dν
end

# %% Set up the reference (twin) experiment
res = "10km"
cd(joinpath(@__DIR__, res))
config_fn = "./enzyme_config_fwd.yml"
arch = CUDABackend()

ν_true  = 200.0     # viscosity that generated the observations
ν_guess = 400.0     # starting guess (the config value); optimizer should recover ν_true

gp = GyreProblem(config_fn, arch)
println("Model: $(gp.nsteps) forward steps per evaluation")

println("=== Generating reference solution at ν★ = $ν_true ===")
h_ref, u_ref = forward_state(gp, ν_true)

# %% Optimization.jl problem
# p is unused (the reference is captured in the closure); the decision variable is
# x[1] = ν. Optimization.jl calls f(x, p) for the objective and grad!(g, x, p)
# for the in-place gradient — both served by a single forward-mode sweep.
function objective(x, p)
    J, _ = loss_and_grad(gp, x[1], h_ref, u_ref)
    return J
end

function objective_grad!(g, x, p)
    _, dJ = loss_and_grad(gp, x[1], h_ref, u_ref)
    g[1] = dJ
    return nothing
end

optf = OptimizationFunction(objective; grad = objective_grad!)
x0   = [ν_guess]
# Physical viscosity is non-negative; bound it well away from zero to keep the
# gyre numerically stable, and cap it above the guess.
prob = OptimizationProblem(optf, x0; lb = [1.0], ub = [1000.0])

println("=== Optimizing (LBFGS) from ν₀ = $ν_guess ===")
# Record the optimizer's trajectory: the Optimization.jl callback fires once per
# accepted iterate with `state.u` (current ν) and `state.objective` (current J).
# Returning false lets the solve continue.
ν_hist = Float64[]
J_hist = Float64[]
function record_history(state, loss_val)
    push!(ν_hist, state.u[1])
    push!(J_hist, loss_val)
    return false
end

# With the O(1) rescaled gradient, a tight g_tol now corresponds to a genuinely
# converged ν rather than tripping on floating-point-scale gradients.
sol = solve(prob, Optim.LBFGS(); g_tol = 1e-6, x_abstol = 1e-4,
            callback = record_history, show_trace = true)

ν_est = sol.u[1]
println()
@show ν_true ν_guess ν_est
@show sol.objective
println("recovery error: $(abs(ν_est - ν_true)) (",
        round(100 * abs(ν_est - ν_true) / ν_true, digits = 4), "% )")

# %% Plots
# ---------------------------------------------------------------------------
# Read mesh coordinates once (cell centres for layerThickness, edges for
# normalVelocity), and evaluate the final model state at the initial guess and
# the recovered viscosity so we can compare fields side by side.
mesh_ds = NCDataset("./initial_state.nc")
xc = Array(mesh_ds["xCell"][:]) ./ 1e3;  yc = Array(mesh_ds["yCell"][:]) ./ 1e3
xe = Array(mesh_ds["xEdge"][:]) ./ 1e3;  ye = Array(mesh_ds["yEdge"][:]) ./ 1e3
close(mesh_ds)

h_guess, u_guess = forward_state(gp, ν_guess)   # final state at the initial guess
h_est,   u_est   = forward_state(gp, ν_est)      # final state at the recovered ν
# (h_ref, u_ref already hold the reference/target final state at ν★.)

# Surface layer (level 1) of the normal velocity for the spatial maps.
u_ref_s   = u_ref[1, :]
u_guess_s = u_guess[1, :]
u_est_s   = u_est[1, :]

# --- Figure 1: solution comparison (normal velocity) + error maps -----------
# A symmetric colour range shared across the three solution panels so they are
# directly comparable; the error panels get their own (smaller) symmetric range.
function symrange(vs...)
    m = maximum(x -> maximum(abs, x), vs)
    m = m == 0 ? 1e-12 : m
    return (-m, m)
end

fig1 = Figure(size = (1500, 760))
Label(fig1[0, 1:6],
      "Barotropic gyre: final surface normal velocity vs viscosity  " *
      "(ν★=$(ν_true), ν₀=$(ν_guess), ν̂=$(round(ν_est, digits=3)))",
      fontsize = 18, font = :bold)

sol_range = symrange(u_ref_s, u_guess_s, u_est_s)
err_guess = u_guess_s .- u_ref_s
err_est   = u_est_s   .- u_ref_s
err_range = symrange(err_guess)   # scale both error maps to the (larger) guess error

# Top row: the three solutions.
for (j, (title, data)) in enumerate((
        ("Reference  (ν★=$(ν_true))",            u_ref_s),
        ("Initial guess  (ν₀=$(ν_guess))",       u_guess_s),
        ("Recovered  (ν̂=$(round(ν_est,digits=2)))", u_est_s)))
    ax = Axis(fig1[1, 2j-1], title = title, xlabel = "x (km)",
              ylabel = j == 1 ? "y (km)" : "", aspect = 1)
    sc = scatter!(ax, xe, ye; color = data, colormap = :balance,
                  colorrange = sol_range, markersize = 5)
    j == 3 && Colorbar(fig1[1, 6], sc, label = "u (m/s)")
end

# Bottom row: error maps (guess−ref, est−ref) + a numeric summary panel.
for (j, (title, data)) in enumerate((
        ("Error: initial guess − reference", err_guess),
        ("Error: recovered − reference",     err_est)))
    ax = Axis(fig1[2, 2j-1], title = title, xlabel = "x (km)",
              ylabel = j == 1 ? "y (km)" : "", aspect = 1)
    sc = scatter!(ax, xe, ye; color = data, colormap = :balance,
                  colorrange = err_range, markersize = 5)
    j == 2 && Colorbar(fig1[2, 4], sc, label = "Δu (m/s)")
end

rms(v) = sqrt(sum(abs2, v) / length(v))
# Numeric summary panel: no aspect constraint (so the text is not squeezed into a
# square and clipped) and spanning two columns for width.
info = Axis(fig1[2, 5:6]); hidedecorations!(info); hidespines!(info)
xlims!(info, 0, 1); ylims!(info, 0, 1)
text!(info, 0.02, 0.98; align = (:left, :top), fontsize = 16,
      text = @sprintf("""
      RMS velocity error
        initial guess : %.3e m/s
        recovered     : %.3e m/s
        reduction     : %.0f×

      viscosity
        true      ν★ = %.4f
        guess     ν₀ = %.4f
        recovered ν̂ = %.6f
        rel. error   = %.2e
      """,
      rms(err_guess), rms(err_est),
      rms(err_guess) / max(rms(err_est), eps()),
      ν_true, ν_guess, ν_est, abs(ν_est - ν_true) / ν_true))

save(joinpath(@__DIR__, "param_estimation_solution.png"), fig1; px_per_unit = 2)
println("wrote param_estimation_solution.png")

# --- Figure 2: optimization history -----------------------------------------
# Prepend the starting point (iteration 0) so the trajectory begins at ν₀.
J0, _   = loss_and_grad(gp, ν_guess, h_ref, u_ref)
νs_full = vcat(ν_guess, ν_hist)
Js_full = vcat(J0,      J_hist)
iters   = 0:(length(νs_full) - 1)
νerr    = abs.(νs_full .- ν_true)

fig2 = Figure(size = (1500, 460))
Label(fig2[0, 1:3], "Parameter estimation: optimization history",
      fontsize = 18, font = :bold)

# (a) viscosity iterate vs iteration, with the true value marked.
axν = Axis(fig2[1, 1], title = "Viscosity iterate",
           xlabel = "iteration", ylabel = "ν")
hlines!(axν, [ν_true]; color = :black, linestyle = :dash, label = "ν★ (true)")
scatterlines!(axν, iters, νs_full; color = :dodgerblue, label = "ν (LBFGS)")
axislegend(axν; position = :rt)

# (b) objective J vs iteration (log scale) — the residual energy collapsing.
axJ = Axis(fig2[1, 2], title = "Objective  J(ν)  (scaled ×$(Int(LOSS_SCALE)))",
           xlabel = "iteration", ylabel = "J", yscale = log10)
scatterlines!(axJ, iters, max.(Js_full, eps()); color = :crimson)

# (c) |ν − ν★| vs iteration (log scale) — convergence of the estimate.
axE = Axis(fig2[1, 3], title = "Estimate error  |ν − ν★|",
           xlabel = "iteration", ylabel = "|ν − ν★|", yscale = log10)
scatterlines!(axE, iters, max.(νerr, eps()); color = :seagreen)

save(joinpath(@__DIR__, "param_estimation_history.png"), fig2; px_per_unit = 2)
println("wrote param_estimation_history.png")

# --- Figure 3: 1-D objective landscape --------------------------------------
# Since ν is scalar, the whole inverse problem lives on a line: scan J(ν) to show
# the (locally convex) basin, and mark the guess, the true value, and where the
# optimizer landed.
ν_scan = collect(range(max(1.0, ν_true - 250), ν_guess + 100; length = 25))
J_scan = [first(loss_and_grad(gp, ν, h_ref, u_ref)) for ν in ν_scan]

fig3 = Figure(size = (760, 560))
axL = Axis(fig3[1, 1], title = "Objective landscape  J(ν)",
           xlabel = "viscosity ν", ylabel = "J(ν)  (scaled ×$(Int(LOSS_SCALE)))",
           yscale = log10)
lines!(axL, ν_scan, max.(J_scan, eps()); color = :gray30)
scatter!(axL, ν_scan, max.(J_scan, eps()); color = :gray30, markersize = 6)
# Trajectory of the optimizer, overlaid on the landscape.
scatterlines!(axL, νs_full, max.(Js_full, eps());
              color = :dodgerblue, markersize = 9, label = "LBFGS path")
vlines!(axL, [ν_true];  color = :black,   linestyle = :dash, label = "ν★ (true)")
vlines!(axL, [ν_guess]; color = :orange,  linestyle = :dot,  label = "ν₀ (guess)")
vlines!(axL, [ν_est];   color = :crimson, linestyle = :dashdot, label = "ν̂ (recovered)")
axislegend(axL; position = :lt)

save(joinpath(@__DIR__, "param_estimation_landscape.png"), fig3; px_per_unit = 2)
println("wrote param_estimation_landscape.png")

# %% The twin experiment recovers ν★ to ~1e-8 relative in practice (exact
# forward-mode gradient on a smooth, locally convex objective). 1e-3 is a
# comfortable, non-flaky bound well inside that.
@test isapprox(ν_est, ν_true, rtol = 1e-3)
