using CUDA: @allowscalar
using KernelAbstractions

mycopyto!(dest, src) = copyto!(dest, src)

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CPU())
    # global i = 0
    # Run the model 
    while !isRinging(simulationAlarm)
        advance!(clock)
        # global i += 1
        ocn_timestep(timestep, Prog, Diag, Tend, Setup, ForwardEuler; backend=backend)
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct... unless we want to apply AD
            reset!(outputAlarm)
        end
    end

    return nothing
end

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CUDABackend())
    # Run the model
    while !isRinging(simulationAlarm)
        advance!(clock)
        ocn_timestep(timestep, Prog, Diag, Tend, Setup, ForwardEuler; backend=backend)
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct... unless we want to apply AD
            reset!(outputAlarm)
        end
    end
    
    sumKernel! = sumArray(backend, 1)
    sumKernel!(sumGPU, Prog.ssh[end], size(Prog.ssh[end])[1], ndrange=1)

    mycopyto!(sumCPU, sumGPU)
    return sumCPU[1]
    
end

@kernel function sumArray(sumGPU, array, arrayLength)
    for j = 1:arrayLength
        sumGPU[1] = sumGPU[1] + array[j]*array[j]
    end
end