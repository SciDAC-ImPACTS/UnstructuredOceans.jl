module MOKAEnzymeExt

using MOKA
using Enzyme
import KernelAbstractions as KA
import MOKA: MPASMesh, Clock, OneTimeAlarm, PeriodicAlarm

# These `inactive_type` declarations tell Enzyme that the listed types carry no
# differentiable data. They are general AD-correctness guards (not specific to any
# one example) and must be in scope whenever MOKA's forward model is differentiated.
# Shipping them in this package extension means any user who loads both MOKA and
# Enzyme gets them automatically, instead of each script having to re-declare them.

# KA Kernel objects (backend + function ref) are not differentiable; marking them
# inactive forces Enzyme to treat kernel locals as Const, which is required for
# KA's own augmented_primal/reverse rules (func::Const{<:Kernel}) to fire instead
# of Enzyme tracing into CUDA kernel compilation code that involves Module types.
Enzyme.EnzymeRules.inactive_type(::Type{<:KA.Kernel}) = true

# Mesh geometry is never differentiable. These inactive_type declarations cover
# the full type hierarchy so Enzyme's static analysis never treats mesh-derived
# CuArrays as potentially-active. Without this, two failure modes occur:
#   1. EnzymeRuntimeActivityError: Enzyme sees a Const mutable struct's CuArray
#      heap pointer stored to the callee's stack (!enzymejl_byref_MUT_REF).
#   2. GPU segfault at synchronize: runtime activity analysis incorrectly activates
#      mesh CuArrays that are annotated @Const in GPU kernels; reverse kernels then
#      try to write gradients to read-only GPU memory.
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Mesh})         = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.HorzMesh})     = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.VerticalMesh}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.PrimaryCells}) = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.DualCells})    = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.Edges})        = true
Enzyme.EnzymeRules.inactive_type(::Type{<:MPASMesh.ActiveLevels}) = true

# Clock and alarm types contain DateTime / Dict / Period fields that are not
# differentiable.  Enzyme.make_zero produces invalid shadows for them and can
# corrupt the backward tape, triggering GPU segfaults.  Mark the whole type
# hierarchy inactive; clock/alarms are pure loop-control, not model state.
Enzyme.EnzymeRules.inactive_type(::Type{<:Clock})         = true
Enzyme.EnzymeRules.inactive_type(::Type{<:OneTimeAlarm})  = true
Enzyme.EnzymeRules.inactive_type(::Type{<:PeriodicAlarm}) = true

# Checkpointed reverse-mode adjoint of the forward model. See the stub +
# rationale in src/forward/run_loop.jl: differentiating the whole multi-timestep
# loop in one `autodiff` call keeps every step's Enzyme tape resident in the CUDA
# in-kernel malloc heap and overflows it past a handful of steps. Here we store
# one state checkpoint per step on the forward sweep, then run the reverse sweep
# one timestep at a time so each tape is freed between steps.
#
# The loss is sumGPU[1] = Σ ssh[end]² (MOKA.sumArray). Its gradient is seeded
# analytically (MOKA.seed_ssh_cotangent!) rather than by differentiating the sum.
# d_Prog is the running cotangent carried backward across steps and is NOT zeroed
# between them; d_Diag/d_Tend ARE zeroed each step (Diag/Tend are recomputed fresh
# every step, so their shadows must not accumulate stale contributions).
function MOKA.ocn_run_loop_checkpointed!(sumGPU, d_sumGPU, timestep, d_timestep,
                                         Prog, d_Prog, Diag, d_Diag, Tend, d_Tend,
                                         Mesh, integrator,
                                         clock, simulationAlarm, outputAlarm)
    backend = KA.get_backend(Prog.ssh[end])

    # --- Forward sweep: snapshot the entering state of each step, then advance.
    # The checkpoint is Prog.{...}[end] as it stands BEFORE ocn_timestep; both
    # integrators' first action is advance_time_levels! ([end] -> [end-1]), so
    # restoring [end] alone reconstructs the exact input to the step.
    checkpoints = NamedTuple[]
    while !MOKA.isRinging(simulationAlarm)
        MOKA.advance!(clock)
        push!(checkpoints, (ssh            = copy(Prog.ssh[end]),
                            normalVelocity = copy(Prog.normalVelocity[end]),
                            layerThickness = copy(Prog.layerThickness[end])))
        MOKA._ad_timestep!(timestep, Prog, Diag, Tend, Mesh, integrator)
        if MOKA.isRinging(outputAlarm)
            MOKA.reset!(outputAlarm)
        end
    end

    # --- Loss value (for logging/inspection) from the final ssh.
    n_ssh = length(Prog.ssh[end])
    sumKernel! = MOKA.sumArray(backend, 1)
    sumKernel!(sumGPU, Prog.ssh[end], n_ssh, ndrange=1)

    # --- Seed the reverse sweep: d_Prog.ssh[end] = 2*ssh[end]*d_sumGPU.
    seedKernel! = MOKA.seed_ssh_cotangent!(backend, 64)
    seedKernel!(d_Prog.ssh[end], Prog.ssh[end], d_sumGPU, n_ssh, ndrange=n_ssh)

    # --- Reverse sweep: one timestep at a time, newest to oldest.
    for k in length(checkpoints):-1:1
        cp = checkpoints[k]
        Prog.ssh[end]            .= cp.ssh
        Prog.normalVelocity[end] .= cp.normalVelocity
        Prog.layerThickness[end] .= cp.layerThickness

        Enzyme.make_zero!(d_Diag)
        Enzyme.make_zero!(d_Tend)

        autodiff(Enzyme.Reverse, MOKA._ad_timestep!,
            Duplicated(timestep, d_timestep),
            Duplicated(Prog, d_Prog),
            Duplicated(Diag, d_Diag),
            Duplicated(Tend, d_Tend),
            Const(Mesh),
            Const(integrator))
    end

    return nothing
end

end # module MOKAEnzymeExt
