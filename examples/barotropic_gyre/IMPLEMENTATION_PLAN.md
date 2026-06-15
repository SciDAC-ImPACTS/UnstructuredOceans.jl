# Barotropic Gyre — Solver Implementation Plan

Goal: make MOKA actually integrate the barotropic gyre so a western boundary
current / clockwise gyre develops, matching the Polaris tutorial.

The example inputs (`setup.py`, `config.cfg`, `Moka.yaml`, `analysis.jl`,
`plot.jl`) are already in place. What follows is the **solver** work required,
since the gyre exercises code paths MOKA does not yet support.

All line numbers refer to the state of the repo at the time of writing.

---

## 0. Prerequisite blocker: non-periodic meshes with solid walls

The gyre mesh is non-periodic (`is_periodic = NO`) and has solid walls.
Empirically (100 km culled mesh, `nCells = 168`): **102 of 555 edges are
boundary edges** with a `0` entry in `cellsOnEdge`. MOKA currently assumes
periodic, stacked meshes:

- `VerticalMesh(mesh_fp, ...)` hard-errors unless `is_periodic == "YES"`
  (`src/infra/MPASMesh/VertMesh.jl:50`).
- Velocity/diagnostic kernels index `cellsOnEdge[1/2, iEdge]` directly, so a
  boundary edge dereferences cell index `0` → out-of-bounds / garbage.
- There is no no-slip wall enforcement.

Everything below depends on fixing this first.

---

## 1. Allow non-periodic meshes

**File:** `src/infra/MPASMesh/VertMesh.jl:50`

Change the hard `error(...)` for `is_periodic != "YES"` into a warning (or a
check that simply proceeds). The stacked-column assertion
(`maxLevelCell .== nVertLevels`) is still satisfied for the culled gyre mesh
(all cells active, single level), so it can stay.

---

## 2. Boundary-edge mask + no-slip enforcement

**File:** `src/infra/MPASMesh/HorzMesh.jl`

1. Add a field to the `Edges` @kwdef struct (line ~64):
   ```julia
   boundaryEdge::IV   # 1 if the edge touches a missing (0) cell, else 0
   ```
2. In `readEdgeInfo` (line ~247) compute it after reading `cellsOnEdge`:
   ```julia
   boundaryEdge = Int32.(vec(any(cellsOnEdge .== 0; dims=1)))
   ```
   and pass it to the `Edges(...)` constructor.
3. Add `boundaryEdge = Adapt.adapt(to, edges.boundaryEdge)` to
   `Adapt.adapt_structure(to, edges::Edges)` (line ~357).

No-slip / no-normal-flow is then enforced implicitly: `normalVelocity` is
initialised to 0 (see `setup.py`) and every velocity-tendency kernel below
skips boundary edges, so wall-edge velocity stays exactly 0.

---

## 3. Guard the kernels that index `cellsOnEdge`

Add an early `boundaryEdge[iEdge] == 1 && return` (or `continue` in the level
loop) to each kernel that dereferences `cellsOnEdge`. Pass `boundaryEdge` as a
new kernel argument from the corresponding launcher.

| Kernel | File | Why |
|---|---|---|
| `SSHGradOnEdge!` | `src/ocn/Tendencies/normalVelocity/pressure_gradient.jl:45` | indexes `cellsOnEdge[1/2]` |
| `interpolateCell2Edge` | `src/ocn/Operators.jl:199` | builds `layerThicknessEdge`; indexes `cellsOnEdge[1/2]` |
| `GradientOnEdge` | `src/ocn/Operators.jl:82` | indexes `cellsOnEdge[1/2]` (if used by gyre path) |
| del2 kernel | `src/ocn/Tendencies/normalVelocity/horizontal_momentum_mixing.jl` | indexes `cellsOnEdge`, `verticesOnEdge` |
| wind forcing kernel | new file (§4) | indexes `cellsOnEdge` for interp |

`DivergenceOnCell` (`Operators.jl:21`) and `CurlOnVertex` (`Operators.jl:120`)
iterate `edgesOnCell` / `edgesOnVertex` (all valid) and rely on boundary-edge
`normalVelocity = 0`, so they need **no** change — zero wall velocity ⇒ zero
flux through walls, which is the correct no-normal-flow condition.

The coriolis kernel (`horizontal_advection_and_coriolis.jl:50`) already skips
`eoe == 0`; leave as is.

---

## 4. Wind-stress forcing

### 4a. Read & project onto edges
**File:** `src/infra/MPASMesh/HorzMesh.jl`

- Add `windForcingEdge::FV` to `Edges` (edge-normal wind acceleration / ρ,
  units m² s⁻²; the per-step `/h` happens in the kernel).
- In `readEdgeInfo`, if `windStressZonal`/`windStressMeridional` are present
  (they are, on cells, written by `setup.py`), interpolate to edges via the two
  `cellsOnEdge` neighbours and project onto the edge normal with `angleEdge`:
  ```julia
  τz_e = 0.5(τz[c1] + τz[c2]);  τm_e = 0.5(τm[c1] + τm[c2])   # skip if c==0
  windForcingEdge = (τz_e*cos(angleEdge) + τm_e*sin(angleEdge)) / rho
  ```
  Default to `zeros(nEdges)` when the fields are absent (keeps the periodic
  IGW case a no-op).
- Thread a `rho` kwarg through `readEdgeInfo` → `ReadHorzMesh` →
  `ocn_setup_mesh`, read from `config.cfg`/namelist (default 1000.0).
- Update `Adapt.adapt_structure(::Edges)`.

### 4b. Tendency
**New file:** `src/ocn/Tendencies/normalVelocity/wind_forcing.jl`

Surface-layer only:
```
tendNormalVelocity[1, iEdge] += windForcingEdge[iEdge] / layerThicknessEdge[1, iEdge]
```
Skip boundary edges. `layerThicknessEdge` comes from `Diag` (already computed
in `diagnostic_compute!`). `include` it in `normalVelocity.jl` and call it from
`computeNormalVelocityTendency!`.

---

## 5. Fix and wire del2 momentum mixing

**File:** `src/ocn/Tendencies/normalVelocity/horizontal_momentum_mixing.jl`

Current bugs to fix:
- launcher calls `SSHGradOnEdge!` instead of the del2 kernel
  `horizontalm_momentum_mixing_del2`; wire the correct kernel.
- `macLevelEdge.Top` → `maxLevelEdge.Top`.
- `dcEdgeInv = 1.0 / dcEdgeInv[iEdge]` is self-referential → use `dcEdge[iEdge]`
  (and `dvEdge[iEdge]`).
- `viscDel2 = 1.0` hard-coded → take viscosity from config (see below).

Operator (standard MPAS del2, both terms already available in `Diag`):
```
tend[k,e] += ν · ( (divᶜ[k,c2] − divᶜ[k,c1]) / dcEdge[e]
                  − (ζᵛ[k,v2]  − ζᵛ[k,v1])  / dvEdge[e] )
```
`velocityDivCell` and `relativeVorticity` are computed in
`diagnostic_compute!`. Skip boundary edges (§3).

Viscosity source: read `config_mom_del2` from the `hmix_del2` namelist in
`ocn_setup_mesh` and stash it as a scalar `momentumDel2::Float64` field on the
`Edges` struct (concrete type → no new type parameter), defaulting to `0.0`.
With `ν = 0` the del2 term vanishes, so the IGW case is unaffected even if the
kernel is always launched.

Wire the call into `computeNormalVelocityTendency!`
(`src/ocn/Tendencies/normalVelocity/normalVelocity.jl:21`) after the coriolis
term, passing `Diag` (for `velocityDivCell`/`relativeVorticity`) and the
viscosity.

---

## 6. Config threading summary

`ocn_setup_mesh` (`src/forward/init.jl:43`) already has `Config`. Read there:
- `hmix_del2 → config_mom_del2`  → `momentumDel2`
- density (add `rho` to `Moka.yaml`, or hard-default 1000.0) → wind projection

Pass both into `ReadHorzMesh(mesh_fp; backend, momentumDel2, rho)` →
`readEdgeInfo`.

---

## 7. Time integrator / config consistency

The driver (`src/driver/mpas_ocean.jl:40`) and `run.sh` always use
**ForwardEuler**, regardless of `config_time_integrator` (the RK4 path in
`time_integration.jl:58` references a non-existent `computeTendency!` and
`Diag.restingThickness`, i.e. it is currently broken). Therefore:

- Set `config_time_integrator: Forward-Euler` in `Moka.yaml`.
- Forward-Euler + wind/del2 needs a small `config_dt` for stability
  (the CFL `dt_max = min(0.25·Δx/2u, 0.25/f₀)`; for Δx=10 km, f₀=1e-3 ⇒
  dt_max ≈ 250 s, so use dt ≈ 50 s). A 2-year spin-up at 50 s is ~1.3 M steps —
  expect a long run; consider a coarser resolution or shorter duration for a
  first smoke test.

(Separately, the RK4 path could be repaired, but it is out of scope here.)

---

## 8. Enzyme considerations

New `Edges` fields (`boundaryEdge`, `windForcingEdge`, `momentumDel2`) are
covered by the existing `inactive_type(::Type{<:MPASMesh.Edges})` declaration
in `examples/inertial_gravity_wave/enzyme_test.jl`, so they stay `Const` and do
not perturb the AD path. Forcing and viscosity are parameters, not state, so
zero gradient is correct.

---

## 9. Validation (on a compute node — cannot run on the login node)

1. `python setup.py --res 10 --dir 10km` → check `initial_state.nc` has
   `windStressZonal`, `layerThickness`, `normalVelocity`, `fCell/Edge/Vertex`.
2. Short smoke run (e.g. a few hundred steps) to confirm no boundary crashes and
   that a circulation spins up (non-zero `normalVelocity`, SSH gradient).
3. Full spin-up, then `julia analysis.jl` → compare numerical vs. Munk
   streamfunction, expect a western-intensified gyre and a modest L2 error.
4. Regression-check the IGW case still runs and the Enzyme test still passes
   (forcing/del2 are no-ops there).

---

## File-change checklist

- [ ] `src/infra/MPASMesh/VertMesh.jl` — allow non-periodic (§1)
- [ ] `src/infra/MPASMesh/HorzMesh.jl` — `boundaryEdge`, `windForcingEdge`,
      `momentumDel2`, projection in `readEdgeInfo`, adapt (§2, §4a, §5)
- [ ] `src/ocn/Operators.jl` — boundary guard in `interpolateCell2Edge`,
      `GradientOnEdge` (§3)
- [ ] `src/ocn/Tendencies/normalVelocity/pressure_gradient.jl` — boundary guard (§3)
- [ ] `src/ocn/Tendencies/normalVelocity/horizontal_momentum_mixing.jl` —
      fix bugs + viscosity + boundary guard (§5)
- [ ] `src/ocn/Tendencies/normalVelocity/wind_forcing.jl` — new tendency (§4b)
- [ ] `src/ocn/Tendencies/normalVelocity/normalVelocity.jl` — wire wind + del2 (§4b, §5)
- [ ] `src/forward/init.jl` — read `config_mom_del2`/`rho`, thread to mesh (§6)
- [ ] `examples/barotropic_gyre/Moka.yaml` — Forward-Euler + dt (§7)
