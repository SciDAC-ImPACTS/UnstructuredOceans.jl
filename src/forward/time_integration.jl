# define our parent abstract type
abstract type timeStepper end
# define the supported timeStepper types to dispatch on.
abstract type ForwardEuler <: timeStepper end
abstract type RungeKutta4  <: timeStepper end

using CUDA: @allowscalar
using KernelAbstractions

function advanceTimeLevels!(Prog::PrognosticVars; backend=CUDABackend())

    nthreads = 100

    kernel2d! = advance_2d_array(backend, nthreads)
    kernel3d! = advance_3d_array(backend, nthreads)

    for field_name in propertynames(Prog)

        ndim = field_name == :ssh ? 1 : 2

        field = getproperty(Prog, field_name)

        if length(field) > 2 error("nTimeLevels must be <= 2") end

        # Here: set first entry of Vector{Array} equal to second

        # some short hand for this would be nice
        if ndim == 1
            #field[:,end-1] .= field[:,end]
            #@show size(field), size(field)[1]
            kernel2d!(field[1], field[2], size(field[1])[1], ndrange=size(field[1])[1])
        else
            #field[:,:,end-1] .= field[:,:,end]
            #kernel3d!(field, ndrange=(size(field)[1],size(field)[2]))
            kernel3d!(field[1], field[2], size(field[1])[2], ndrange=size(field[1])[2])
        end

        setproperty!(Prog, field_name, field)
    end
end

@kernel function advance_2d_array(fieldPrev, @Const(fieldNext), arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds fieldPrev[j] = fieldNext[j]
    end
    @synchronize()
end

@kernel function advance_3d_array(fieldPrev, @Const(fieldNext), arrayLength)
    #i, j = @index(Global, NTuple)
    #@inbounds fieldPrev[i, j] = fieldNext[i, j]

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds fieldPrev[1, j] = fieldNext[1, j]
    end
    @synchronize()
end

function ocn_timestep(Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      S::ModelSetup,
                      ::Type{RungeKutta4};
                      backend = KA.CPU())

    Mesh = S.mesh
    Clock = S.timeManager

    # advance the timelevels within the state strcut
    advanceTimeLevels!(Prog; backend=backend)

    # convert the timestep to seconds
    dt = convert(Float64, Dates.value(Second(Clock.timeStep)))

    a = [dt/2., dt/2., dt]
    b = [dt/6., dt/3., dt/3., dt/6.]

    # lets assume that we've already swapped time dimensions so that the
    # end-1 position is the "current" timestep and the "end" position can be
    # the "next" timestep, which itself is actually the substeps of the RK
    # method.
    #
    #@views begin
    #    normalVelocityCurr = Prog.normalVelocity[:,:,end-1]
    #    layerThicknessCurr = Prog.layerThickness[:,:,end-1]
    #    normalVelocityProvis = Prog.normalVelocity[:,:,end]
    #    layerThicknessProvis = Prog.layerThickness[:,:,end]
    #end

    sshCurr = @view Prog.ssh[:,end-1]
    normalVelocityCurr = @view Prog.normalVelocity[:,:,end-1]
    layerThicknessCurr = @view Prog.layerThickness[:,:,end-1]

    sshProvis = @view Prog.ssh[:,end]
    normalVelocityProvis = @view Prog.normalVelocity[:,:,end]
    layerThicknessProvis = @view Prog.layerThickness[:,:,end]

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # this will be the t+1 timestep, i.e. it's the array the rk4 updates are
    # accumulated into, not this is NOT a view b/c that would have the substeps
    # being overwritten byt the accumulate step.
    #normalVelocityNew = normalVelocity[:,:,end-1]
    sshNew = ssh[:,end]
    normalVelocityNew = normalVelocity[:,:,end]
    layerThicknessNew = layerThickness[:,:,end]

    for RK_step in 1:4
        # compute tenedencies using the provis state
        computeTendency!(Mesh, Diag, Prog, Tend, :normalVelocity)
        computeTendency!(Mesh, Diag, Prog, Tend, :layerThickness)

        # unpack the tendecies for updating the substep state.
        @unpack tendNormalVelocity, tendLayerThickness = Tend

        # update the substep state which is storred in the final time postion
        # of the Prog structure
        if RK_step < 4

            normalVelocityProvis .= normalVelocityCurr .+ a[RK_step] .* tendNormalVelocity
            layerThicknessProvis .= layerThicknessCurr .+ a[RK_step] .* tendLayerThickness
            # compute ssh from layerThickness
            sshProvis = layerThicknessProvis .- sum(Mesh.VertMesh.restingThickness; dims=1)
            # compute the diagnostics using the Provis State,
            # i.e. the substage solution
            diagnostic_compute!(Mesh, Diag, Prog)
        end

        # accumulate the update in the NEW time position array
        normalVelocityNew .= normalVelocityNew .+ b[RK_step] .* tendNormalVelocity
        layerThicknessNew .= layerThicknessNew .+ b[RK_step] .* tendLayerThickness
        sshNew = layerThicknessNew .- sum(Diag.restingThickness; dims=1)
    end

    # place the NEW solution in the appropriate location in the Prog arrays
    normalVelocity[:,:,end] = normalVelocityNew
    layerThickness[:,:,end] = layerThicknessNew

    # put the updated solution back in the Prog strcutre
    @pack! Prog = ssh, normalVelocity, layerThickness

    ## compute diagnostics for new state
    diagnostic_compute!(Mesh, Diag, Prog)
end

function ocn_timestep(timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Forcing::ForcingVars,
                      S::ModelSetup,
                      ::Type{ForwardEuler};
                      backend = CUDABackend())

    Mesh = S.mesh
    Clock = S.timeManager
    Config = S.config

    # advance the timelevels within the state strcut
    advanceTimeLevels!(Prog; backend=backend)

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog; backend = backend)

    # compute normalVelocity tenedency
    computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh, Forcing, Config;
                                   backend = backend)

    # compute layerThickness tendency
    computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh, Config;
                                   backend = backend)

    # update the state variables by the tendencies
    nthreads = 50
    tendKernel! = UpdateStateVariable!(backend, nthreads)

    tendKernel!(normalVelocity[end], Tend.tendNormalVelocity, timestep, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)
    tendKernel!(layerThickness[end], Tend.tendLayerThickness, timestep, Mesh.HorzMesh.PrimaryCells.nCells, ndrange=Mesh.HorzMesh.PrimaryCells.nCells)

    ssh_length = size(ssh[end])[1]

    kernel! = Update_ssh!(backend, nthreads)
    kernel!(ssh[end], Prog.layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)

    @pack! Prog = ssh, normalVelocity, layerThickness

end

# Zeros out a vector along its entire length
@kernel function UpdateStateVariable!(var, @Const(tendVar), @Const(dt), arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        var[1,j] = var[1,j] + dt[1] * tendVar[1, j]
    end
    @synchronize()
end


@kernel function Update_ssh!(ssh, @Const(layerThickness), @Const(restingThicknessSum), arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds ssh[j] = layerThickness[1,j] - restingThicknessSum[j]
    end
    @synchronize()
end
