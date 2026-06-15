using CUDA: @allowscalar
using KernelAbstractions

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm;
                      backend=CPU(), print_interval=500)
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

    while !isRinging(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, ForwardEuler; backend=backend)
        step += 1

        if step % print_interval == 0
            elapsed = time() - t_start
            rate    = step / elapsed
            eta     = (total_steps - step) / rate
            pct     = 100.0 * step / total_steps
            @printf("  step %8d / %d  (%5.1f%%)  sim=%-20s  wall=%6.0fs  %8.1f steps/s  ETA %.0fs\n",
                    step, total_steps, pct,
                    Dates.format(clock.currTime, "yyyy-mm-dd HH:MM:SS"),
                    elapsed, rate, eta)
        end

        if isRinging(outputAlarm)
            reset!(outputAlarm)
        end
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
function ocn_run_loop(sumGPU, timestep, Prog, Diag, Tend, Mesh, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CUDABackend())
    while !isRinging(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, ForwardEuler; backend=backend)
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