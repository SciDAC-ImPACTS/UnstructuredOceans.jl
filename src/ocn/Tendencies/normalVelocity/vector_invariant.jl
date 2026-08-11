"""
Vector-invariant (nonlinear) momentum advection + Coriolis, following the MPAS
TRiSK discretization. Selected by config as an alternative to `linearCoriolis`:

```math
\\frac{\\partial u_e}{\\partial t} \\mathrel{+}=
    \\underbrace{\\sum_{e'} w_{e,e'}\\, F_{e'}\\, \\bar q_{e,e'}}_{\\text{PV flux}}
    \\;-\\; \\underbrace{\\left[\\nabla K\\right]_e}_{\\text{KE gradient}}
```

with ``F = h\\,u`` the thickness flux (already diagnosed), ``q = (f+\\zeta)/h`` the
potential vorticity at vertices, ``\\bar q`` its edge average, and ``K`` the
kinetic energy on cells. The `linearCoriolis` term is the ``h\\to`` const,
``\\zeta\\to 0`` limit of the PV flux (``\\sum w\\,f\\,u``); this term adds the
nonlinear advection that case omits.

The reconstructions (KE on cells, thickness and PV on vertices) are stored in
[`DiagnosticVars`](@ref) and filled here, so the default `linearCoriolis` path and
`diagnostic_compute!` are untouched.
"""
abstract type vectorInvariant <: Coriolis end

function horizontal_advection_and_coriolis_tendency!(Tend::TendencyVars,
                                                     Prog::PrognosticVars,
                                                     Diag::DiagnosticVars,
                                                     Mesh::Mesh,
                                                     ::Type{vectorInvariant};
                                                     nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack nVertLevels, maxLevelEdge = VertMesh
    @unpack nEdges, nEdgesOnEdge, weightsOnEdge, edgesOnEdge = Edges
    @unpack dcEdge, dvEdge, cellsOnEdge, verticesOnEdge, boundaryEdge = Edges
    @unpack nCells, nEdgesOnCell, edgesOnCell, areaCell = PrimaryCells
    @unpack nVertices, vertexDegree, cellsOnVertex, areaTriangle = DualCells
    @unpack fᵛ, kiteAreasOnVertex = DualCells

    normalVelocity = Prog.normalVelocity[end]
    layerThickness = Prog.layerThickness[end]

    @unpack thicknessFlux, relativeVorticity = Diag
    @unpack kineticEnergyCell, layerThicknessVertex, potentialVorticityVertex = Diag
    @unpack tendNormalVelocity = Tend

    # 1. Reconstructions ------------------------------------------------------
    ke! = kinetic_energy_cell!(backend, nthreads)
    ke!(kineticEnergyCell, normalVelocity, nEdgesOnCell, edgesOnCell,
        dcEdge, dvEdge, areaCell, nCells, ndrange=(nCells, nVertLevels))

    hv! = layer_thickness_vertex!(backend, nthreads)
    hv!(layerThicknessVertex, layerThickness, cellsOnVertex, kiteAreasOnVertex,
        areaTriangle, vertexDegree, nVertices, ndrange=(nVertices, nVertLevels))

    pv! = potential_vorticity_vertex!(backend, nthreads)
    pv!(potentialVorticityVertex, relativeVorticity, layerThicknessVertex, fᵛ,
        nVertices, ndrange=(nVertices, nVertLevels))

    # 2. PV-flux Coriolis/advection ------------------------------------------
    pvflux! = pv_flux_tendency_kernel!(backend, nthreads)
    pvflux!(tendNormalVelocity, thicknessFlux, potentialVorticityVertex,
            nEdgesOnEdge, edgesOnEdge, weightsOnEdge, verticesOnEdge,
            maxLevelEdge.Top, ndrange=nEdges)

    # 3. Kinetic-energy gradient ---------------------------------------------
    keg! = kinetic_energy_gradient_kernel!(backend, nthreads)
    keg!(tendNormalVelocity, kineticEnergyCell, cellsOnEdge, dcEdge,
         boundaryEdge, maxLevelEdge.Top, ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

# Kinetic energy on cells: K_i = (1/A_i) Σ_{e∈EC(i)} (1/8) · dc_e · dv_e · u_e².
# The per-edge area ¼·dc·dv sums to the cell area (Σ ¼ dc dv / A = 1), so the ⅛
# coefficient makes K = ½|u|² for a uniform velocity field (the MPAS convention).
@kernel function kinetic_energy_cell!(kineticEnergyCell,
                                      @Const(normalVelocity),
                                      @Const(nEdgesOnCell),
                                      @Const(edgesOnCell),
                                      @Const(dcEdge),
                                      @Const(dvEdge),
                                      @Const(areaCell),
                                      arrayLength)
    iCell, k = @index(Global, NTuple)
    if iCell < arrayLength + 1
        @inbounds invArea = 1.0 / areaCell[iCell]
        acc = 0.0
        @inbounds for i in 1:nEdgesOnCell[iCell]
            iEdge = edgesOnCell[i, iCell]
            u = normalVelocity[k, iEdge]
            acc += 0.125 * dcEdge[iEdge] * dvEdge[iEdge] * u * u
        end
        @inbounds kineticEnergyCell[k, iCell] = acc * invArea
    end
end

# Layer thickness interpolated to vertices via kite areas:
# h_v = (1/A_v) Σ_{i} kiteArea_{i,v} · h_{cell(i,v)}. Missing (0) cells skipped.
@kernel function layer_thickness_vertex!(layerThicknessVertex,
                                         @Const(layerThickness),
                                         @Const(cellsOnVertex),
                                         @Const(kiteAreasOnVertex),
                                         @Const(areaTriangle),
                                         vertexDegree,
                                         arrayLength)
    iVertex, k = @index(Global, NTuple)
    if iVertex < arrayLength + 1
        @inbounds invArea = 1.0 / areaTriangle[iVertex]
        acc = 0.0
        @inbounds for i in 1:vertexDegree
            iCell = cellsOnVertex[i, iVertex]
            if iCell != 0
                acc += kiteAreasOnVertex[i, iVertex] * layerThickness[k, iCell]
            end
        end
        @inbounds layerThicknessVertex[k, iVertex] = acc * invArea
    end
end

# Potential vorticity at vertices: q_v = (f_v + ζ_v) / h_v. Guard the boundary
# case h_v == 0 (a vertex with no active surrounding cells) → q = 0.
@kernel function potential_vorticity_vertex!(potentialVorticityVertex,
                                             @Const(relativeVorticity),
                                             @Const(layerThicknessVertex),
                                             @Const(fᵛ),
                                             arrayLength)
    iVertex, k = @index(Global, NTuple)
    if iVertex < arrayLength + 1
        @inbounds hv = layerThicknessVertex[k, iVertex]
        if hv > 0.0
            @inbounds potentialVorticityVertex[k, iVertex] =
                (fᵛ[iVertex] + relativeVorticity[k, iVertex]) / hv
        else
            @inbounds potentialVorticityVertex[k, iVertex] = 0.0
        end
    end
end

# PV-flux Coriolis/advection: tend[e] += Σ_{e'} w_{e,e'} · F_{e'} · q̄_{e,e'},
# with F the thickness flux and q̄ = ½(q_e + q_{e'}) the edge-averaged PV, each
# edge PV being ½ of its two vertices. Generalizes coriolis_force_tendency_kernel!
# (which is the h→const, ζ→0 special case Σ w f u).
@kernel function pv_flux_tendency_kernel!(tendency,
                                          @Const(thicknessFlux),
                                          @Const(potentialVorticityVertex),
                                          @Const(nEdgesOnEdge),
                                          @Const(edgesOnEdge),
                                          @Const(weightsOnEdge),
                                          @Const(verticesOnEdge),
                                          @Const(maxLevelEdgeTop))
    iEdge = @index(Global, Linear)

    @inbounds nLevels = maxLevelEdgeTop[iEdge]
    @inbounds v1 = verticesOnEdge[1, iEdge]
    @inbounds v2 = verticesOnEdge[2, iEdge]

    @inbounds for i in 1:nEdgesOnEdge[iEdge]
        eoe = edgesOnEdge[iEdge, i]
        # Structured branch (not `continue`) for Enzyme reverse-mode correctness,
        # and to avoid index-0 loads on boundary edges (see coriolis kernel note).
        if eoe != 0
            @inbounds w    = weightsOnEdge[iEdge, i]
            @inbounds ve1  = verticesOnEdge[1, eoe]
            @inbounds ve2  = verticesOnEdge[2, eoe]
            @inbounds for k in 1:nLevels
                # edge PV = mean of its two vertices; q̄ = mean of the two edges'
                q_e  = 0.5 * (potentialVorticityVertex[k, v1]  + potentialVorticityVertex[k, v2])
                q_ep = 0.5 * (potentialVorticityVertex[k, ve1] + potentialVorticityVertex[k, ve2])
                qbar = 0.5 * (q_e + q_ep)
                tendency[k, iEdge] += w * thicknessFlux[k, eoe] * qbar
            end
        end
    end
end

# Kinetic-energy gradient: tend[e] -= (K_{c2} - K_{c1}) / dc_e. Same structure as
# the SSH pressure gradient; boundary edges skipped.
@kernel function kinetic_energy_gradient_kernel!(tendency,
                                                 @Const(kineticEnergyCell),
                                                 @Const(cellsOnEdge),
                                                 @Const(dcEdge),
                                                 @Const(boundaryEdge),
                                                 @Const(maxLevelEdgeTop))
    iEdge = @index(Global, Linear)
    if boundaryEdge[iEdge] != 1
        @inbounds jCell1 = cellsOnEdge[1, iEdge]
        @inbounds jCell2 = cellsOnEdge[2, iEdge]
        @inbounds invDc = 1.0 / dcEdge[iEdge]
        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k, iEdge] -= invDc *
                (kineticEnergyCell[k, jCell2] - kineticEnergyCell[k, jCell1])
        end
    end
end
