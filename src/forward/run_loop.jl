using CUDA: @allowscalar
using KernelAbstractions

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CPU())
    Mesh = Setup.mesh
    while !isRinging(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Mesh, ForwardEuler; backend=backend)
        if isRinging(outputAlarm)
            reset!(outputAlarm)
        end
    end

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