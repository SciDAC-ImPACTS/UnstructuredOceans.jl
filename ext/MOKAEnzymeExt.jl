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

end # module MOKAEnzymeExt
