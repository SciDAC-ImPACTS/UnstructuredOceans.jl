# define our parent abstract type
abstract type timeStepper end
# define the supported timeStepper types to dispatch on.
abstract type ForwardEuler <: timeStepper end
abstract type RungeKutta4  <: timeStepper end

"""
    parse_integrator(name) -> timeStepper type

Map a `config_time_integrator` string to its timeStepper type, tolerating common
spellings (e.g. "RK4" for RungeKutta4, "Forward-Euler" for ForwardEuler).
"""
function parse_integrator(name::AbstractString)
    key = lowercase(replace(name, "-" => "", "_" => "", " " => ""))
    if key in ("rungekutta4", "rk4")
        return RungeKutta4
    elseif key in ("forwardeuler", "euler", "fe")
        return ForwardEuler
    else
        throw(ArgumentError(
            "Unknown config_time_integrator \"$name\"; " *
            "expected \"RungeKutta4\" (\"RK4\") or \"ForwardEuler\"."))
    end
end

using GPUArraysCore: @allowscalar
using KernelAbstractions

function advanceTimeLevels!(Prog::PrognosticVars)
    backend = KernelAbstractions.get_backend(first(Prog.ssh))

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

@kernel function advance_2d_array(fieldPrev, fieldNext, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds fieldPrev[j] = fieldNext[j]
    end
    @synchronize()
end

@kernel function advance_3d_array(fieldPrev, fieldNext, arrayLength)
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
                      ::Type{RungeKutta4})
    backend = KernelAbstractions.get_backend(Prog.ssh[end])

    Mesh  = S.mesh
    Clock = S.timeManager

    advanceTimeLevels!(Prog)

    dt = convert(Float64, Dates.value(Second(Clock.timeStep)))

    # RK4 coefficients: substep sizes and accumulation weights
    a = [dt/2., dt/2., dt]
    b = [dt/6., dt/3., dt/3., dt/6.]

    # After advanceTimeLevels!, both [end-1] and [end] hold y_n.
    # [end-1] is the read-only current state; [end] is scratch for substeps.
    normalVelocityCurr   = Prog.normalVelocity[end-1]   # Matrix (nVertLevels × nEdges)
    layerThicknessCurr   = Prog.layerThickness[end-1]   # Matrix (nVertLevels × nCells)

    @unpack ssh, normalVelocity, layerThickness = Prog

    # Accumulators start at y_n; each stage adds b[k]*k_i so the final
    # value is y_{n+1} = y_n + dt/6*(k1 + 2k2 + 2k3 + k4).
    normalVelocityNew   = copy(normalVelocity[end-1])
    layerThicknessNew   = copy(layerThickness[end-1])

    nthreads   = 50
    ssh_kernel! = Update_ssh!(backend, nthreads)
    ssh_length  = length(ssh[end])

    diagnostic_compute!(Mesh, Diag, Prog)

    for RK_step in 1:4
        computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh)
        computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh)

        @unpack tendNormalVelocity, tendLayerThickness = Tend

        if RK_step < 4
            normalVelocity[end] .= normalVelocityCurr .+ a[RK_step] .* tendNormalVelocity
            layerThickness[end] .= layerThicknessCurr .+ a[RK_step] .* tendLayerThickness
            ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)
            diagnostic_compute!(Mesh, Diag, Prog)
        end

        normalVelocityNew .+= b[RK_step] .* tendNormalVelocity
        layerThicknessNew .+= b[RK_step] .* tendLayerThickness
    end

    normalVelocity[end] .= normalVelocityNew
    layerThickness[end] .= layerThicknessNew
    ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)

    @pack! Prog = ssh, normalVelocity, layerThickness

    diagnostic_compute!(Mesh, Diag, Prog)
end

function ocn_timestep(timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Mesh::Mesh,
                      ::Type{ForwardEuler})
    backend = KernelAbstractions.get_backend(Prog.ssh[end])

    # advance the timelevels within the state strcut
    advanceTimeLevels!(Prog)

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog)

    # compute normalVelocity tenedency
    computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh)

    # compute layerThickness tendency
    computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh)

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
@kernel function UpdateStateVariable!(var, tendVar, dt, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        var[1,j] = var[1,j] + dt[1] * tendVar[1, j]
    end
    @synchronize()
end


@kernel function Update_ssh!(ssh, layerThickness, restingThicknessSum, arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds ssh[j] = layerThickness[1,j] - restingThicknessSum[j]
    end
    @synchronize()
end