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

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
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

    while !isRinging(simulationAlarm)
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

        if isRinging(outputAlarm)
            if output_ds !== nothing
                frame += 1
                io_writeTimestep(output_ds, Setup, Prog, frame)
                last_written = clock.currTime
            end
            reset!(outputAlarm)
        end
    end

    # Capture the final state if it wasn't already written at a checkpoint
    if output_ds !== nothing && clock.currTime != last_written
        frame += 1
        io_writeTimestep(output_ds, Setup, Prog, frame)
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
    while !isRinging(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, integrator)
        if isRinging(outputAlarm)
            reset!(outputAlarm)
        end
    end

    sumKernel! = sumArray(backend, 1)
    sumKernel!(sumGPU, Prog.ssh[end], size(Prog.ssh[end])[1], ndrange=1)

    return nothing
end

@kernel function sumArray(sumGPU, array, arrayLength)
    for j = 1:arrayLength
        sumGPU[1] = sumGPU[1] + array[j]*array[j]
    end
end

# Bundle of all state advanced by the forward model, packaged as a single mutable struct so it can be checkpointed by Checkpointing.jl.
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

# Advance the model in place by one timestep.
function ocn_step!(model::OceanModel{TS}) where {TS<:timeStepper}
    ocn_timestep(model.dt, model.Prog, model.Diag, model.Tend, model.Mesh, TS)
    return nothing
end

# Loss and loop methods for checkpointed reverse-mode AD.
function ocn_loss end
function ocn_run_loop_checkpointed! end