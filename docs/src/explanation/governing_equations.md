# Governing equations

MOKA integrates the **shallow-water equations** for layer thickness ``h`` and
horizontal velocity ``\mathbf{v}``, under boundary conditions set by the problem:

```math
\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{v}) = 0 ,
```

```math
\frac{\partial \mathbf{v}}{\partial t}
  + (\mathbf{v}\cdot\nabla)\mathbf{v} + f\,\mathbf{k}\times\mathbf{v}
  = -g\nabla h + \nu_2 \nabla^2 \mathbf{v} + \frac{\boldsymbol{\tau}}{\rho h} .
```

The terms on the right-hand side of the momentum equation are, in order: the
**pressure gradient** (with gravitational acceleration ``g``), **Laplacian
(del2) viscosity** (``\nu_2`` the horizontal eddy viscosity), and **wind-stress
forcing** (``\boldsymbol{\tau}`` the surface wind stress, ``\rho`` a reference
density). ``f`` is the Coriolis parameter.

Each term maps to a piece of code in the tendency modules under
`src/ocn/Tendencies/`: the SSH pressure gradient, the (linear) Coriolis term,
the del2 momentum mixing, and the wind forcing, summed in
`compute_normal_velocity_tendency!`; the thickness-flux divergence is assembled in
`compute_layer_thickness_tendency!`. These are advanced in time by
[`ocn_timestep`](@ref).

## Vector-invariant form

The nonlinear and Coriolis terms are written in vector-invariant form,

```math
(\mathbf{v}\cdot\nabla)\mathbf{v} + f\,\mathbf{k}\times\mathbf{v}
  = q\,\mathbf{k}\times(h\mathbf{v}) + \nabla K ,
\qquad
q = \frac{\zeta + f}{h}, \quad K = \frac{|\mathbf{v}|^2}{2} ,
```

with ``q`` the potential vorticity, ``\zeta = \mathbf{k}\cdot(\nabla\times
\mathbf{v})`` the relative vorticity, and ``K`` the kinetic energy. The
prognostic velocity variable is the **edge-normal component** ``v_e``; tangential
and cell-centered reconstructions are diagnosed from it. This mirrors the
MPAS-Ocean formulation.

!!! note "Current physics scope"
    The active dynamical core wires up the pressure gradient, **linear** Coriolis,
    del2 viscosity, and wind forcing. The full nonlinear advection is present in
    the code structure but not yet exercised.

## The two verification cases

MOKA ships two configurations with analytic references, taken from the E3SM
Polaris tasks so the initial and forcing conditions match an established
implementation.

### Inertial gravity wave

On an ``f``-plane, a small perturbation ``\eta`` about a resting depth ``H`` (so
``h = H + \eta``) linearizes the equations to a wave with dispersion relation
``\omega = \sqrt{f^2 + gH(k_x^2 + k_y^2)}`` and a known exact solution on a
doubly-periodic domain. Because the initial amplitude is deliberately small, the
convergence study measures the **spatial discretization error** rather than the
amplitude error. This case exercises the core operators without boundaries or
forcing.

### Barotropic gyre

A closed basin with solid walls on a ``\beta``-plane
(``f = f_0 + \beta y``), forced by a zonal wind stress

```math
\tau_x(y) = -\tau_0 \cos\!\left(\frac{\pi y}{L_y}\right), \qquad \tau_y = 0 .
```

Sverdrup balance in the interior is closed by a frictional western boundary
current — a **Munk layer** of width ``\delta_M = \frac{2\pi}{\sqrt 3}
(\nu_2/\beta)^{1/3}``. This case exercises everything the wave does not: wind
forcing, lateral viscosity, solid boundaries, and a ``\beta``-plane. The steady
state is verified against the analytic no-slip Munk streamfunction.

See [TRiSK discretization](@ref) for how these operators are represented on the
unstructured mesh, and [Getting started](@ref) to run the gyre.
