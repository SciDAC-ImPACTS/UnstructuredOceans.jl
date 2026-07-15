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

function advance_time_levels!(Prog::PrognosticVars; nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(first(Prog.ssh))

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

function ocn_timestep(dt,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Mesh::Mesh,
                      ::Type{RungeKutta4};
                      nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(Prog.ssh[end])

    advance_time_levels!(Prog; nthreads=nthreads)

    # RK4 substep fractions of dt (a) and accumulation weights (b). These are pure
    # HOST Float64 constants; `dt` stays a device array so the update kernels read
    # dt[1] on-device — mirroring forward_euler_step!. Building host coefficient
    # arrays out of the device `dt` instead (e.g. a = [dt/2, dt/2, dt]) and using
    # them in fused broadcasts forces a host↔device copy (cuMemcpyHtoDAsync) that
    # Enzyme's reverse mode cannot differentiate ("unsupported gc-transition tag").
    a = (0.5, 0.5, 1.0)
    b = (1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0)

    # After advance_time_levels!, both [end-1] and [end] hold y_n.
    # [end-1] is the read-only current state; [end] is scratch for substeps.
    normalVelocityCurr   = Prog.normalVelocity[end-1]   # Matrix (nVertLevels × nEdges)
    layerThicknessCurr   = Prog.layerThickness[end-1]   # Matrix (nVertLevels × nCells)

    @unpack ssh, normalVelocity, layerThickness = Prog

    nEdges      = Mesh.HorzMesh.Edges.nEdges
    nCells      = Mesh.HorzMesh.PrimaryCells.nCells
    ssh_length  = length(ssh[end])

    substep!    = rk4_substep!(backend, nthreads)
    accumulate! = rk4_accumulate!(backend, nthreads)
    copy!       = rk4_copy!(backend, nthreads)
    ssh_kernel! = update_sea_surface_height!(backend, nthreads)

    # Accumulators start at y_n; each stage adds b[k]*dt*k_i so the final value is
    # y_{n+1} = y_n + dt/6*(k1 + 2k2 + 2k3 + k4). Kept as their own device arrays so
    # the substep scratch in normalVelocity[end] doesn't clobber them. Filled with a
    # device kernel (not Base.copy) because Enzyme reverse-mode lowers the generic
    # copyto! through a host staging buffer (cuMemcpyHtoDAsync), which it then cannot
    # differentiate (unsupported gc-transition tag).
    normalVelocityNew   = similar(normalVelocityCurr)
    layerThicknessNew   = similar(layerThicknessCurr)
    copy!(normalVelocityNew, normalVelocityCurr, nEdges, ndrange=nEdges)
    copy!(layerThicknessNew, layerThicknessCurr, nCells, ndrange=nCells)

    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)

    for RK_step in 1:4
        computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)
        computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

        @unpack tendNormalVelocity, tendLayerThickness = Tend

        if RK_step < 4
            substep!(normalVelocity[end], normalVelocityCurr, tendNormalVelocity, dt, nEdges, Val(a[RK_step]), ndrange=nEdges)
            substep!(layerThickness[end], layerThicknessCurr, tendLayerThickness, dt, nCells, Val(a[RK_step]), ndrange=nCells)
            ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)
            diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)
        end

        accumulate!(normalVelocityNew, tendNormalVelocity, dt, nEdges, Val(b[RK_step]), ndrange=nEdges)
        accumulate!(layerThicknessNew, tendLayerThickness, dt, nCells, Val(b[RK_step]), ndrange=nCells)
    end

    copy!(normalVelocity[end], normalVelocityNew, nEdges, ndrange=nEdges)
    copy!(layerThickness[end], layerThicknessNew, nCells, ndrange=nCells)
    ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)

    @pack! Prog = ssh, normalVelocity, layerThickness

    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)
end

# RK4 substep: out[1,j] = base[1,j] + A*dt[1]*tend[1,j]. The stage fraction `A` is a
# COMPILE-TIME constant carried as a `Val` type parameter, not a runtime scalar arg:
# Enzyme's KernelAbstractions reverse rule rejects active Float64 kernel arguments
# ("Active kernel arguments not supported on GPU"), so — like forward_euler_step! —
# the only numeric runtime arg is `dt`, a length-1 device array read on-device.
@kernel function rk4_substep!(out, base, tend, dt, arrayLength, ::Val{A}) where {A}
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds out[1, j] = base[1, j] + A * dt[1] * tend[1, j]
    end
    @synchronize()
end

# RK4 accumulation: acc[1,j] += B*dt[1]*tend[1,j]; weight `B` is a compile-time Val.
@kernel function rk4_accumulate!(acc, tend, dt, arrayLength, ::Val{B}) where {B}
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds acc[1, j] = acc[1, j] + B * dt[1] * tend[1, j]
    end
    @synchronize()
end

# Copy of the accumulated y_{n+1} back into the [end] time level: dst[1,j] = src[1,j].
@kernel function rk4_copy!(dst, src, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds dst[1, j] = src[1, j]
    end
    @synchronize()
end

function ocn_timestep(timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Mesh::Mesh,
                      ::Type{ForwardEuler};
                      nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(Prog.ssh[end])

    # advance the timelevels within the state strcut
    advance_time_levels!(Prog; nthreads=nthreads)

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)

    # compute normalVelocity tenedency
    computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

    # compute layerThickness tendency
    computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

    # update the state variables by the tendencies
    tendKernel! = forward_euler_step!(backend, nthreads)

    tendKernel!(normalVelocity[end], Tend.tendNormalVelocity, timestep, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)
    tendKernel!(layerThickness[end], Tend.tendLayerThickness, timestep, Mesh.HorzMesh.PrimaryCells.nCells, ndrange=Mesh.HorzMesh.PrimaryCells.nCells)
    
    ssh_length = size(ssh[end])[1]

    kernel! = update_sea_surface_height!(backend, nthreads)
    kernel!(ssh[end], Prog.layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)
    
    @pack! Prog = ssh, normalVelocity, layerThickness
    
end

# Forward Euler step
@kernel function forward_euler_step!(var, tendVar, dt, arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        var[1,j] = var[1,j] + dt[1] * tendVar[1, j]
    end
    @synchronize()
end


@kernel function update_sea_surface_height!(ssh, layerThickness, restingThicknessSum, arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds ssh[j] = layerThickness[1,j] - restingThicknessSum[j]
    end
    @synchronize()
end