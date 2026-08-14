# Profile the reverse-mode (checkpointed AD) step to quantify WHERE the GPU time
# goes — specifically the device-side malloc / allocation fraction of the Enzyme tape.
# =====================================================================================
# Motivation: on the barotropic gyre the forward model is ~300-15000x faster on the
# GPU than reverse mode, and GPU reverse LOSES to the CPU (see
# resolution_scaling_ad_benchmark.png). The reverse-only scaling exponent is
# super-linear (p~1.23) vs linear on CPU, pointing at device-side tape allocation
# rather than launch latency. This script measures that fraction directly.
#
# It runs a checkpointed reverse sweep twice: once to warm up (compilation), once
# under CUDA.@profile, then prints the host-side profile grouped so the malloc /
# cuMemAlloc / free time is separable from actual compute kernels.
#
# Uses the 40km gyre (1020 cells): smallest mesh, fits in modest free memory, and the
# super-linear reverse overhead is present at every size. Pin the device with
# CUDA_PROFILE_DEVICE (default 0). Requires ~512 MiB free for the AD device heap.
#
# Run:  julia --project examples/barotropic_gyre/profile_reverse_step.jl
# Output is written to examples/barotropic_gyre/profile_reverse_step.log via the
# queue wrapper; here it just prints to stdout.

using Dates
import KernelAbstractions as KA
using Enzyme
using Checkpointing
using UnstructuredOceans
using CUDA
import CUDA: @allowscalar
import UnstructuredOceans.MPASMesh

const DEV = parse(Int, get(ENV, "CUDA_PROFILE_DEVICE", "0"))
CUDA.device!(DEV)
println("Profiling on CUDA device $DEV: ", CUDA.name(CUDA.device()))
# Free-memory readout is best-effort: the accessor name varies across CUDA.jl versions.
try
    free, total = CUDA.memory_info()
    println("Free / total mem: ", Base.format_bytes(free), " / ", Base.format_bytes(total))
catch
    println("Free / total mem: (unavailable on this CUDA.jl)")
end

# Number of steps to differentiate. Small keeps the profile readable; the per-step
# malloc pattern is identical regardless of horizon (checkpointing frees each tape
# between steps), so a short run is representative.
const NSTEPS = parse(Int, get(ENV, "PROFILE_NSTEPS", "8"))
const NSNAPS = parse(Int, get(ENV, "PROFILE_NSNAPS", "4"))

const RES = get(ENV, "PROFILE_RES", "40km")
cd(joinpath(@__DIR__, RES))
const CONFIG = isfile("enzyme_config.yml") ? "enzyme_config.yml" : "config.yml"
println("Config: ", joinpath(RES, CONFIG), "  nsteps=$NSTEPS  nsnaps=$NSNAPS")

backend = CUDABackend()

# Enzyme reverse stores its per-thread tape in device malloc'd buffers; the default
# ~8 MB CUDA heap overflows at model scale. Raise it before differentiating.
set_ad_device_heap!(backend)

function build_model(backend)
    Setup, Diag, Tend, Prog = ocn_init(CONFIG, backend=backend)
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
    integrator = parse_integrator(
        UnstructuredOceans.config_get(UnstructuredOceans.config_get(Setup.config.namelist, "time_integration"),
                       "config_time_integrator"))
    model   = OceanModel(integrator, Prog, Diag, Tend, Setup.mesh, timestep)
    d_model = OceanModel(integrator,
                         Enzyme.make_zero(Prog),
                         Enzyme.make_zero(Diag),
                         Enzyme.make_zero(Tend),
                         Setup.mesh,
                         Enzyme.make_zero(timestep))
    return model, d_model
end

function reverse_run()
    model, d_model = build_model(backend)
    ocn_run_loop_checkpointed!(model, d_model, NSTEPS;
                               mode = Enzyme.Reverse, nsnaps = NSNAPS, verbose = 0)
    CUDA.synchronize()
    return nothing
end

println("\n=== Warmup (compilation) ===")
@time reverse_run()

# The @profile macro returns a results object that only auto-prints in the REPL; under
# stdout redirection we must display it explicitly. NOTE: the Enzyme tape uses in-KERNEL
# device malloc from the fixed heap, so that cost lands INSIDE each kernel's runtime in
# the device trace (not as a separate cuMemAlloc API row) — the device kernel summary is
# therefore the view that matters. Each block is wrapped so one API mismatch can't abort
# the whole (expensive) run.
show_profile(label, res) = (println("\n=== $label ==="); show(stdout, MIME("text/plain"), res); println())

try
    res_trace = CUDA.@profile reverse_run()
    show_profile("CUDA.@profile: full device trace (per-launch timeline)", res_trace)
catch e
    println("\n[full trace failed: $e]")
end

try
    res_summary = CUDA.@profile trace=false reverse_run()
    show_profile("CUDA.@profile trace=false: grouped kernel summary (time by name)", res_summary)
catch e
    println("\n[grouped summary failed: $e]")
end

println("\nDone.")
