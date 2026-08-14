"""
    Constants(gravity, density)
    Constants(; gravity=9.80616, density=1000.0, backend=KA.CPU())

Physical constants built once from configuration and carried on the [`Mesh`](@ref)
so kernels read them instead of hardcoding literals.

Each scalar is stored as a **1-element device array** (not a bare `Float64`), the
same convention as the edge viscosity `momentumDel2`: a by-value `Float64` kernel
argument is classified *Active* by Enzyme under reverse-mode AD ("Active kernel
arguments not supported on GPU"), whereas a 1-element array sourced from the
(inactive) `Mesh` is treated as `Const`. `Adapt.adapt` moves the struct between
host and device.

Fields:
- `gravity`  — gravitational acceleration ``g`` ``[\\mathrm{m\\,s^{-2}}]``.
- `density`  — reference density ``\\rho_0`` ``[\\mathrm{kg\\,m^{-3}}]``.
"""
struct Constants{FV}
    gravity::FV
    density::FV
end

function Constants(; gravity::Real = 9.80616, density::Real = 1000.0,
                     backend = KA.CPU())
    # Build 1-element host arrays, then move to the requested backend (mirrors how
    # readEdgeInfo wraps momentumDel2). Avoids scalar indexing into a device array.
    g = Adapt.adapt(backend, [Float64(gravity)])
    ρ = Adapt.adapt(backend, [Float64(density)])
    return Constants(g, ρ)
end

function Adapt.adapt_structure(backend, x::Constants)
    return Constants(Adapt.adapt(backend, x.gravity),
                     Adapt.adapt(backend, x.density))
end
