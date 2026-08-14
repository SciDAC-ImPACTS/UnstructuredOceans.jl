using GPUArraysCore: @allowscalar
using KernelAbstractions

# Thresholds beyond which the simulation is considered diverged.
const _SSH_MAX  = 1e4   # [m]   — unreachable by any physical barotropic gyre
const _VEL_MAX  = 1e3   # [m/s] — well above any ocean speed

function _check_diverged(Prog, step, sim_time)
    ssh  = Prog.ssh[end]
    vel  = Prog.normalVelocity[end]
    h    = Prog.layerThickness[end]

    max_ssh = maximum(abs, ssh)
    max_vel = maximum(abs, vel)
    min_h   = minimum(h)

    diverged = false
    if any(isnan, ssh) || any(isnan, vel)
        @printf("  DIVERGED at step %d (%s): NaN detected\n", step, sim_time)
        diverged = true
    elseif any(isinf, ssh) || any(isinf, vel)
        @printf("  DIVERGED at step %d (%s): Inf detected\n", step, sim_time)
        diverged = true
    elseif max_ssh > _SSH_MAX
        @printf("  DIVERGED at step %d (%s): |ssh|_max = %.3e m exceeds threshold %.3e m\n",
                step, sim_time, max_ssh, _SSH_MAX)
        diverged = true
    elseif max_vel > _VEL_MAX
        @printf("  DIVERGED at step %d (%s): |vel|_max = %.3e m/s exceeds threshold %.3e m/s\n",
                step, sim_time, max_vel, _VEL_MAX)
        diverged = true
    elseif min_h <= 0.0
        @printf("  DIVERGED at step %d (%s): layerThickness_min = %.3e m (non-positive)\n",
                step, sim_time, min_h)
        diverged = true
    end

    return diverged, max_ssh, max_vel
end

"""
    ocn_run_loop(timestep, Prog, Diag, Tend, Setup, integrator,
                 clock, simulationAlarm, outputAlarm;
                 print_interval=500, output_ds=nothing)

Advance the forward model from the current clock time until `simulationAlarm`
rings, integrating with `integrator` ([`RungeKutta4`](@ref) or
[`ForwardEuler`](@ref)).

This is the top-level driver of a simulation. Each iteration advances the
[`Clock`](@ref) by one step, calls [`ocn_timestep`](@ref) to update `Prog` in
place, and — whenever `outputAlarm` rings — appends a frame to `output_ds` via
[`io_write_timestep`](@ref). Progress (throughput, ETA, and field magnitudes) is
printed every `print_interval` steps, and the run aborts early if the state
diverges (NaN/Inf, or `ssh`/velocity/thickness leaving physical bounds).

Arguments:
- `timestep` — the step size as a length-1 device array of `Float64` seconds
  (kept on-device so the update kernels read it without a host copy).
- `Prog`, `Diag`, `Tend` — the state from [`ocn_init`](@ref).
- `Setup` — the `ModelSetup` (provides the `Mesh`).
- `clock`, `simulationAlarm`, `outputAlarm` — from [`ocn_init_alarms`](@ref).

Keyword arguments:
- `print_interval` — steps between progress lines.
- `output_ds` — an open `NCDataset` from [`io_initialize`](@ref), or `nothing`
  to run without writing output.

Returns `nothing`; `Prog` holds the final state on return.
"""
function ocn_run_loop(timestep, Prog, Diag, Tend, Setup, integrator, clock, simulationAlarm, outputAlarm;
                      print_interval=500, output_ds=nothing)
    Mesh = Setup.mesh

    total_ms    = Dates.value(Millisecond(simulationAlarm.ringTime - clock.startTime))
    dt_ms       = Dates.value(Millisecond(clock.timeStep))
    total_steps = total_ms ÷ dt_ms
    step        = 0
    t_start     = time()

    total_secs = total_ms ÷ 1000
    dur_str = @sprintf("%dd %02dh %02dm %02ds",
                       total_secs ÷ 86400, (total_secs % 86400) ÷ 3600,
                       (total_secs % 3600) ÷ 60, total_secs % 60)
    @printf("  Running %d steps (dt = %ds, duration = %s)\n",
            total_steps, dt_ms ÷ 1000, dur_str)

    # frame 1 is the initial state written by io_initialize; next frame starts at 2
    frame        = 1
    last_written = clock.startTime

    while !is_ringing(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, integrator)
        step += 1

        if step % print_interval == 0
            elapsed  = time() - t_start
            rate     = step / elapsed
            eta      = (total_steps - step) / rate
            pct      = 100.0 * step / total_steps
            sim_time = Dates.format(clock.currTime, "yyyy-mm-dd HH:MM:SS")

            diverged, max_ssh, max_vel = _check_diverged(Prog, step, sim_time)
            diverged && break

            @printf("  step %8d / %d  (%5.1f%%)  sim=%-20s  wall=%5.0fs  %7.1f steps/s  ETA %6.0fs  |ssh|=%8.2e  |vel|=%8.2e\n",
                    step, total_steps, pct, sim_time,
                    elapsed, rate, eta, max_ssh, max_vel)
        end

        if is_ringing(outputAlarm)
            if output_ds !== nothing
                frame += 1
                io_write_timestep(output_ds, Setup, Prog, frame)
                last_written = clock.currTime
            end
            reset!(outputAlarm)
        end
    end

    # Capture the final state if it wasn't already written at a checkpoint
    if output_ds !== nothing && clock.currTime != last_written
        frame += 1
        io_write_timestep(output_ds, Setup, Prog, frame)
    end

    elapsed = time() - t_start
    @printf("  Done: %d steps in %.1fs (%.1f steps/s)\n", step, elapsed, step / elapsed)

    return nothing
end

# AD variant: takes Mesh directly to keep ModelSetup (and its yaml_config mutable
# struct fields) out of the differentiated scope, avoiding EnzymeRuntimeActivityError.
# sumCPU is intentionally absent: copyto!(cpu, gpu) lowers to cuMemcpyDtoHAsync_v2
# which carries a gc-transition LLVM operand bundle that Enzyme's GradientUtils
# does not support. The caller must pre-seed d_sumGPU and do the D2H copy outside autodiff.
function ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Mesh, integrator, clock, simulationAlarm, outputAlarm)
    backend = KernelAbstractions.get_backend(Prog.ssh[end])
    while !is_ringing(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, integrator)
        if is_ringing(outputAlarm)
            reset!(outputAlarm)
        end
    end

    sumKernel! = sum_array(backend, 1)
    sumKernel!(sumGPU, Prog.ssh[end], size(Prog.ssh[end])[1], ndrange=1)

    return nothing
end

@kernel function sum_array(sumGPU, array, arrayLength)
    @inbounds for j = 1:arrayLength
        sumGPU[1] = sumGPU[1] + array[j]*array[j]
    end
end

"""
    ocn_run_loop_fwd!(viscDel2, timestep, Prog, Diag, Tend, Mesh, integrator, nsteps)

Advance the model `nsteps` fixed-size steps for **forward-mode** automatic
differentiation with respect to the Laplacian viscosity.

`viscDel2` is threaded (as a length-1 device array) into every
[`ocn_timestep`](@ref), so when this is called under
`Enzyme.autodiff(Forward, ...)` with `viscDel2` as `Duplicated`, the shadow of
`Prog` on return holds ∂(final state)/∂(viscosity). Forward mode needs no scalar
loss, tape, or checkpointing, so it avoids the device-heap growth of the
reverse-mode path (contrast [`ocn_run_loop_checkpointed!`](@ref)). `Mesh` is
passed directly to keep the mutable `ModelSetup`/config out of the differentiated
scope. `Prog` is advanced in place; returns `nothing`.
"""
function ocn_run_loop_fwd!(viscDel2, timestep, Prog, Diag, Tend, Mesh, integrator, nsteps::Int)
    for _ in 1:nsteps
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, integrator; viscDel2=viscDel2)
    end
    return nothing
end

"""
    OceanModel{TS<:timeStepper}(Prog, Diag, Tend, Mesh, dt)

All state advanced by the forward model, bundled into a single mutable struct so
it can be checkpointed by [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl).

The time integrator is carried as the type parameter `TS` (e.g.
`OceanModel(RungeKutta4, Prog, Diag, Tend, Mesh, dt)`) so [`ocn_step!`](@ref) can
dispatch on it without storing a runtime field. `dt` is the step size as a
length-1 device array. Used as the evolving unit of state in the checkpointed
reverse-mode adjoint; see [`ocn_run_loop_checkpointed!`](@ref).
"""
mutable struct OceanModel{TS<:timeStepper, P, D, T, M, V}
    Prog::P
    Diag::D
    Tend::T
    Mesh::M
    dt::V
end

function OceanModel(::Type{TS}, Prog, Diag, Tend, Mesh, dt) where {TS<:timeStepper}
    OceanModel{TS, typeof(Prog), typeof(Diag), typeof(Tend), typeof(Mesh), typeof(dt)}(
        Prog, Diag, Tend, Mesh, dt)
end

"""
    ocn_step!(model::OceanModel)

Advance an [`OceanModel`](@ref) in place by one timestep, using the integrator
carried in its type parameter. This single-step form is the unit the checkpointed
adjoint differentiates one step at a time; returns `nothing`.
"""
function ocn_step!(model::OceanModel{TS}) where {TS<:timeStepper}
    ocn_timestep(model.dt, model.Prog, model.Diag, model.Tend, model.Mesh, TS)
    return nothing
end

# Loss and loop methods for checkpointed reverse-mode AD.

"""
    ocn_loss(model::OceanModel, nsteps, scheme) -> Float64

Scalar objective for reverse-mode automatic differentiation: run the model
`nsteps` steps under a Checkpointing.jl `scheme` (Revolve) and return
``J = \\sum_i \\mathrm{ssh}_i^2`` over all cells at the final time.

This is a function stub; the method is provided by the `MOKACheckpointingExt`
extension, which loads automatically when both `Checkpointing` and `Enzyme` are
in scope alongside `MOKA`. Differentiating `ocn_loss` with Enzyme yields the
sensitivity of `J` to the initial state. See the
[Automatic differentiation](@ref) guide.
"""
function ocn_loss end

"""
    ocn_run_loop_checkpointed!(model::OceanModel, nsteps, scheme)

Advance an [`OceanModel`](@ref) `nsteps` steps under a Checkpointing.jl `scheme`
(Revolve), differentiating a single [`ocn_step!`](@ref) at a time so each Enzyme
tape is freed before the next — bounding device-heap use during the reverse
sweep.

This is a function stub; the method is provided by the `MOKACheckpointingExt`
extension (requires `Checkpointing` and `Enzyme` loaded alongside `MOKA`). Before
differentiating on a GPU, raise the in-kernel malloc heap with
[`set_ad_device_heap!`](@ref). See the [Automatic differentiation](@ref) guide.
"""
function ocn_run_loop_checkpointed! end