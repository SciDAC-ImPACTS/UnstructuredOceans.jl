# define our parent abstract type
abstract type timeStepper end

"""
    ForwardEuler <: timeStepper

First-order explicit forward-Euler time integrator. Pass the type (not an
instance) to [`ocn_timestep`](@ref) to select this scheme, or select it by
configuration string via [`parse_integrator`](@ref) (`"ForwardEuler"`,
`"euler"`).
"""
abstract type ForwardEuler <: timeStepper end

"""
    RungeKutta4 <: timeStepper

Classical fourth-order explicit Runge–Kutta (RK4) time integrator. Pass the type
(not an instance) to [`ocn_timestep`](@ref) to select this scheme, or select it
by configuration string via [`parse_integrator`](@ref) (`"RungeKutta4"`,
`"RK4"`). This is the default integrator used in the verification studies.
"""
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
            # 2-D launch over (ncols, nVertLevels) so every level is advanced.
            nlevels = size(field[1])[1]
            ncols   = size(field[1])[2]
            kernel3d!(field[1], field[2], ncols, ndrange=(ncols, nlevels))
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
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds fieldPrev[k, j] = fieldNext[k, j]
    end
    @synchronize()
end

"""
    ocn_timestep(dt, Prog, Diag, Tend, Mesh, integrator;
                 viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
                 nthreads=DEFAULT_NTHREADS)

Advance the prognostic state `Prog` in place by one step of size `dt`, using the
time integrator type `integrator` ([`RungeKutta4`](@ref) or
[`ForwardEuler`](@ref), passed as a type).

Each step recomputes diagnostics with `diagnostic_compute!`, evaluates the
normal-velocity and layer-thickness tendencies, and updates the state via
KernelAbstractions kernels. `dt` is a length-1 device array of seconds; keeping
it on-device (rather than a host `Float64`) is what lets the update kernels be
differentiated by Enzyme. `viscDel2` is the Laplacian viscosity (defaulting to
the value stored on the mesh edges); threading it explicitly enables
forward-mode AD with respect to viscosity. `nthreads` sets the kernel workgroup
size. Returns `nothing`.
"""
function ocn_timestep(dt,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Mesh::Mesh,
                      ::Type{RungeKutta4};
                      coriolis=MOKA.NormalVelocity.linearCoriolis,
                      viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
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
    nVertLevels = Mesh.VertMesh.nVertLevels
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
    copy!(normalVelocityNew, normalVelocityCurr, nEdges, ndrange=(nEdges, nVertLevels))
    copy!(layerThicknessNew, layerThicknessCurr, nCells, ndrange=(nCells, nVertLevels))

    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)

    for RK_step in 1:4
        compute_normal_velocity_tendency!(Tend, Prog, Diag, Mesh; coriolis=coriolis, viscDel2=viscDel2, nthreads=nthreads)
        compute_layer_thickness_tendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

        @unpack tendNormalVelocity, tendLayerThickness = Tend

        if RK_step < 4
            substep!(normalVelocity[end], normalVelocityCurr, tendNormalVelocity, dt, nEdges, Val(a[RK_step]), ndrange=(nEdges, nVertLevels))
            substep!(layerThickness[end], layerThicknessCurr, tendLayerThickness, dt, nCells, Val(a[RK_step]), ndrange=(nCells, nVertLevels))
            ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, nVertLevels, ndrange=ssh_length)
            diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)
        end

        accumulate!(normalVelocityNew, tendNormalVelocity, dt, nEdges, Val(b[RK_step]), ndrange=(nEdges, nVertLevels))
        accumulate!(layerThicknessNew, tendLayerThickness, dt, nCells, Val(b[RK_step]), ndrange=(nCells, nVertLevels))
    end

    copy!(normalVelocity[end], normalVelocityNew, nEdges, ndrange=(nEdges, nVertLevels))
    copy!(layerThickness[end], layerThicknessNew, nCells, ndrange=(nCells, nVertLevels))
    ssh_kernel!(ssh[end], layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, nVertLevels, ndrange=ssh_length)

    @pack! Prog = ssh, normalVelocity, layerThickness

    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)
end

# RK4 substep: out[k,j] = base[k,j] + A*dt[1]*tend[k,j]. The stage fraction `A` is a
# COMPILE-TIME constant carried as a `Val` type parameter, not a runtime scalar arg:
# Enzyme's KernelAbstractions reverse rule rejects active Float64 kernel arguments
# ("Active kernel arguments not supported on GPU"), so — like forward_euler_step! —
# the only numeric runtime arg is `dt`, a length-1 device array read on-device.
# 2-D launch over (ncols, nVertLevels) so every vertical level is updated.
@kernel function rk4_substep!(out, base, tend, dt, arrayLength, ::Val{A}) where {A}
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds out[k, j] = base[k, j] + A * dt[1] * tend[k, j]
    end
    @synchronize()
end

# RK4 accumulation: acc[k,j] += B*dt[1]*tend[k,j]; weight `B` is a compile-time Val.
@kernel function rk4_accumulate!(acc, tend, dt, arrayLength, ::Val{B}) where {B}
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds acc[k, j] = acc[k, j] + B * dt[1] * tend[k, j]
    end
    @synchronize()
end

# Copy of the accumulated y_{n+1} back into the [end] time level: dst[k,j] = src[k,j].
@kernel function rk4_copy!(dst, src, arrayLength)
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds dst[k, j] = src[k, j]
    end
    @synchronize()
end

function ocn_timestep(timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Mesh::Mesh,
                      ::Type{ForwardEuler};
                      coriolis=MOKA.NormalVelocity.linearCoriolis,
                      viscDel2=Mesh.HorzMesh.Edges.momentumDel2,
                      nthreads=DEFAULT_NTHREADS)
    backend = KernelAbstractions.get_backend(Prog.ssh[end])

    # advance the timelevels within the state strcut
    advance_time_levels!(Prog; nthreads=nthreads)

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog; nthreads=nthreads)

    # compute normalVelocity tenedency
    compute_normal_velocity_tendency!(Tend, Prog, Diag, Mesh; coriolis=coriolis, viscDel2=viscDel2, nthreads=nthreads)

    # compute layerThickness tendency
    compute_layer_thickness_tendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

    # update the state variables by the tendencies
    nEdges      = Mesh.HorzMesh.Edges.nEdges
    nCells      = Mesh.HorzMesh.PrimaryCells.nCells
    nVertLevels = Mesh.VertMesh.nVertLevels

    tendKernel! = forward_euler_step!(backend, nthreads)

    tendKernel!(normalVelocity[end], Tend.tendNormalVelocity, timestep, nEdges, ndrange=(nEdges, nVertLevels))
    tendKernel!(layerThickness[end], Tend.tendLayerThickness, timestep, nCells, ndrange=(nCells, nVertLevels))

    ssh_length = size(ssh[end])[1]

    kernel! = update_sea_surface_height!(backend, nthreads)
    kernel!(ssh[end], Prog.layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, nVertLevels, ndrange=ssh_length)
    
    @pack! Prog = ssh, normalVelocity, layerThickness
    
end

# Forward Euler step. 2-D launch over (ncols, nVertLevels) so every level updates.
@kernel function forward_euler_step!(var, tendVar, dt, arrayLength)
    j, k = @index(Global, NTuple)
    if j < arrayLength + 1
        @inbounds var[k,j] = var[k,j] + dt[1] * tendVar[k, j]
    end
    @synchronize()
end


# SSH is the column integral of layer thickness minus the resting column height:
# ssh[j] = Σ_k layerThickness[k,j] - restingThicknessSum[j]. Launched 1-D over cells
# (each thread sums its own column). The nVertLevels loop is a plain sequential sum,
# not a parallel dimension, so this stays a single write per cell (Enzyme-friendly).
# For nVertLevels == 1 the sum is layerThickness[1,j], identical to before.
@kernel function update_sea_surface_height!(ssh, layerThickness, restingThicknessSum, arrayLength, nVertLevels)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        acc = 0.0
        @inbounds for k in 1:nVertLevels
            acc += layerThickness[k, j]
        end
        @inbounds ssh[j] = acc - restingThicknessSum[j]
    end
    @synchronize()
end