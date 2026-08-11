module MOKACheckpointingExt

# Checkpointed reverse-mode AD of MOKA's forward model. This extension loads only
# when BOTH Checkpointing and Enzyme are present (see Project.toml [extensions]).
# The general Enzyme AD guards (inactive_type declarations for mesh/clock/alarm
# types) live in MOKAEnzymeExt, which is already loaded whenever Enzyme is, so they
# are in effect here too.

using MOKA
using Enzyme
using Checkpointing
import KernelAbstractions as KA
import MOKA: OceanModel

# Checkpointed forward loss. See the stub + rationale in src/forward/run_loop.jl:
# differentiating the whole multi-timestep loop in one `autodiff` call keeps every
# step's Enzyme tape resident in the CUDA in-kernel malloc heap and overflows it
# past a handful of steps. The `@ad_checkpoint` macro (Checkpointing.jl) instead
# stores a logarithmic number of state checkpoints and, on the reverse sweep,
# differentiates a SINGLE `ocn_step!` at a time, so each Enzyme tape is freed
# between steps. The loop body touches only `model`, so the closure Checkpointing
# builds captures exactly that one struct (no stray scalar capture).
#
# The loss is sumGPU[1] = Σ ssh[end]², accumulated by MOKA.sum_array after the loop.
# We do NOT differentiate a GPU `sum` reduction (Enzyme has no rule for CUDA's
# mapreduce and errors on it); instead the loss is written by a KA kernel — which
# Enzyme differentiates like every other kernel in `ocn_timestep` — and the reverse
# sweep is seeded by setting the kernel's output shadow `d_sumGPU = 1` (see the
# driver). `ocn_loss` therefore returns `nothing`; the cotangent enters through
# `sumGPU` and Checkpointing carries it back to the initial state in `d_model.Prog`.
function MOKA.ocn_loss(model::OceanModel, sumGPU, scheme::Checkpointing.Scheme, nsteps::Int)
    @ad_checkpoint scheme for i = 1:nsteps
        MOKA.ocn_step!(model)
    end
    backend = KA.get_backend(model.Prog.ssh[end])
    n = length(model.Prog.ssh[end])
    sumKernel! = MOKA.sum_array(backend, 1)
    sumKernel!(sumGPU, model.Prog.ssh[end], n, ndrange = 1)
    return nothing
end

# Run the checkpointed adjoint: advance `model` `nsteps` steps and accumulate the
# gradient of Σ ssh[end]² w.r.t. the initial state into `d_model`. `d_model` must
# be a zeroed shadow of `model` (its `Mesh` may alias the primal, as mesh geometry
# is inactive). `nsnaps` is the number of Revolve checkpoints to retain. Returns
# the scalar loss. The reverse sweep is seeded analytically by `d_sumGPU = 1`: the
# adjoint of sumGPU[1] = Σ ssh² is d_ssh = 2*ssh, applied by the kernel's reverse.
function MOKA.ocn_run_loop_checkpointed!(model::OceanModel, d_model::OceanModel,
                                         nsteps::Int; mode = Enzyme.Reverse, nsnaps::Int = 4, verbose::Int = 0)
    backend  = KA.get_backend(model.Prog.ssh[end])
    sumGPU   = KA.zeros(backend, Float64, 1)
    d_sumGPU = KA.ones(backend, Float64, 1)   # seed d(loss)/d(loss) = 1
    scheme   = Revolve(nsnaps; verbose = verbose)
    autodiff(mode, MOKA.ocn_loss, Const,
             Duplicated(model, d_model),
             Duplicated(sumGPU, d_sumGPU),
             Const(scheme),
             Const(nsteps))
    return Array(sumGPU)[1]
end

end # module MOKACheckpointingExt
