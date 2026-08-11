"""
    ForcingVars(windStressEdge)
    ForcingVars(nEdges; backend=KA.CPU())

External forcing fields carried on the [`Mesh`](@ref) and applied by the
dispatched forcing tendency terms (see `AbstractForcing` / `WindForcing`).

Presently holds the surface wind stress projected onto edge normals and divided
by the reference density, `windStressEdge` ``[\\mathrm{m^2\\,s^{-2}}]`` per unit
thickness, dimensioned `(nEdges,)`. It is built from the forcing stream (see
`ocn_setup_mesh`); when wind forcing is disabled the field is left zero so the
wind tendency is a no-op without touching the differentiated timestep path.

Additional forcings (surface fluxes, restoring, tidal) add fields here and a
matching `AbstractForcing` method. Being part of the (Enzyme-inactive) `Mesh`,
these arrays are read as constants by the differentiated kernels.
`Adapt.adapt` moves the struct between host and device.
"""
struct ForcingVars{FV}
    # edge-normal wind stress / rho [m^2 s^-2 per unit thickness]; dim (nEdges,)
    windStressEdge::FV
end

# Empty (zeroed) forcing — used by the backward-compatible Mesh constructors and
# whenever no forcing stream is configured.
ForcingVars(nEdges::Integer; backend = KA.CPU()) =
    ForcingVars(KA.zeros(backend, Float64, nEdges))

"""
    build_forcing(horzMesh, forcing_fp; rho=1000.0, use_wind_stress=true,
                  backend=KA.CPU()) -> ForcingVars

Build the [`ForcingVars`](@ref) for a horizontal mesh by reading the forcing
stream file `forcing_fp` and projecting its surface wind stress
(`windStressZonal`/`windStressMeridional`, at cell centres) onto the edge normals,
divided by the reference density `rho`. Returns zeroed forcing when
`use_wind_stress=false` or the wind fields are absent — so the wind tendency
becomes a no-op without touching the differentiated timestep path (mirrors the
`momentumDel2 = 0` gate for mixing).

The projection uses the host-side edge geometry (`angleEdge`, `cellsOnEdge`) of
`horzMesh`; the result is moved to `backend`.
"""
function build_forcing(horzMesh, forcing_fp::AbstractString;
                       rho::Float64 = 1000.0, use_wind_stress::Bool = true,
                       backend = KA.CPU())
    edges       = horzMesh.Edges
    nEdges      = edges.nEdges
    angleEdge   = Adapt.adapt(Array, edges.angleEdge)
    cellsOnEdge = Adapt.adapt(Array, edges.cellsOnEdge)

    windStressEdge = zeros(Float64, nEdges)

    if use_wind_stress
        ds = NCDataset(forcing_fp, "r")
        if haskey(ds, "windStressZonal") && haskey(ds, "windStressMeridional")
            τz_c = Float64.(ds["windStressZonal"][:, 1])
            τm_c = Float64.(ds["windStressMeridional"][:, 1])
            for iEdge in 1:nEdges
                c1 = cellsOnEdge[1, iEdge]
                c2 = cellsOnEdge[2, iEdge]
                τz_e = (c1 == 0 ? τz_c[c2] : (c2 == 0 ? τz_c[c1] : 0.5*(τz_c[c1] + τz_c[c2])))
                τm_e = (c1 == 0 ? τm_c[c2] : (c2 == 0 ? τm_c[c1] : 0.5*(τm_c[c1] + τm_c[c2])))
                windStressEdge[iEdge] = (τz_e * cos(angleEdge[iEdge]) +
                                         τm_e * sin(angleEdge[iEdge])) / rho
            end
        end
        close(ds)
    end

    return ForcingVars(Adapt.adapt(backend, windStressEdge))
end

function Adapt.adapt_structure(backend, x::ForcingVars)
    return ForcingVars(Adapt.adapt(backend, x.windStressEdge))
end
