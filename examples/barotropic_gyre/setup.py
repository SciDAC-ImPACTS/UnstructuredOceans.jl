#!/usr/bin/env python3

import argparse
import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from polaris.config import PolarisConfigParser
from polaris.mesh.planar import compute_planar_hex_nx_ny
from polaris.ocean.vertical import init_vertical_coord

# Physics parameters for the Munk (no-slip) barotropic gyre.  Values match the
# Polaris `barotropic_gyre` task, `barotropic_gyre_munk_no-slip` section:
# https://docs.e3sm.org/polaris/main/users_guide/ocean/tasks/barotropic_gyre.html
# The no-slip variant is used here because Moka.jl imposes solid (no-slip) walls.
F_0   = 1.0e-3    # Coriolis parameter at southern boundary [s⁻¹]
BETA  = 1.0e-10   # meridional gradient of Coriolis parameter [s⁻¹ m⁻¹]
TAU_0 = 0.1       # peak zonal wind stress [N m⁻²]
NU_2  = 400.0     # horizontal (Laplacian) viscosity [m² s⁻¹]
RHO   = 1000.0    # reference ocean density [kg m⁻³]
G     = 9.80616   # gravitational acceleration [m s⁻²]
LX    = 1000.0    # domain length in x [km]
LY    = 1000.0    # domain length in y [km]
H     = 5000.0    # resting depth [m]


def munk_layer_width():
    """Anticipated width of the lateral (Munk) boundary layer [m]."""
    return (2.0 * np.pi) / np.sqrt(3.0) * (NU_2 / BETA) ** (1.0 / 3.0)


def exact_streamfunction(x, y, boundary_condition='no-slip',
                         beta=BETA, nu=NU_2):
    """Exact barotropic streamfunction for the linearized Munk experiment.

    Mirrors `Analysis.exact_solution` in the Polaris barotropic_gyre task.
    `x`, `y` are coordinates (already shifted to the domain origin) in metres.
    """
    L_x = float(x.max() - x.min())
    L_y = float(y.max() - y.min())

    delta_m = (nu / (beta * L_y ** 3.0)) ** (1.0 / 3.0)
    gamma = (np.sqrt(3.0) * x) / (2.0 * delta_m * L_x)

    if boundary_condition == 'no-slip':
        psi = np.pi * np.sin(np.pi * y / L_y) * (
            1.0 - (x / L_x)
            - np.exp(-x / (2.0 * delta_m * L_x))
            * (np.cos(gamma)
               + ((1.0 - 2.0 * delta_m) / np.sqrt(3.0)) * np.sin(gamma))
            + delta_m * np.exp(((x / L_x) - 1.0) / delta_m)
        )
    elif boundary_condition == 'free-slip':
        psi = np.pi * np.sin(np.pi * (y / L_y)) * (
            (1.0 - (x / L_x) - delta_m)
            + np.exp((-(x / L_x)) / (2.0 * delta_m))
            * ((-2.0 / 3.0) * (1.0 - delta_m) * np.cos(gamma - (np.pi / 6.0))
               + (2.0 / np.sqrt(3.0)) * np.sin(gamma))
            + delta_m * np.exp((((x / L_x) - 1.0) / delta_m))
        )
    else:
        raise ValueError(f'unknown boundary_condition: {boundary_condition}')

    return psi


def create_initial_state(resolution_km, output_dir):
    """Create initial_state.nc and culled_mesh.nc for one resolution.

    Parameters
    ----------
    resolution_km : float
        Grid cell spacing in kilometres.
    output_dir : str
        Directory where output files are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    dc = resolution_km * 1e3
    nx, ny = compute_planar_hex_nx_ny(LX, LY, resolution_km)

    # Non-periodic mesh: the gyre is bounded by solid walls on all sides
    ds_mesh = make_planar_hex_mesh(
        nx=nx, ny=ny, dc=dc, nonperiodic_x=True, nonperiodic_y=True
    )
    ds_mesh = cull(ds_mesh)
    ds_mesh = convert(ds_mesh)

    write_netcdf(ds_mesh, os.path.join(output_dir, 'culled_mesh.nc'))

    # Beta-plane Coriolis on cells, edges, and vertices: f = f0 + beta * y
    for loc in ['Cell', 'Edge', 'Vertex']:
        ds_mesh[f'f{loc}'] = F_0 + BETA * ds_mesh[f'y{loc}']

    # Vertical grid from config.cfg (ocean.cfg supplies framework defaults)
    config = PolarisConfigParser()
    config.add_from_package('polaris.ocean', 'ocean.cfg')
    config.add_from_file('config.cfg')

    ds = ds_mesh.copy()

    # Ocean at rest: zero SSH, flat bottom
    ds['ssh'] = xr.zeros_like(ds_mesh.xCell)
    ds['bottomDepth'] = H * xr.ones_like(ds_mesh.xCell)

    init_vertical_coord(config, ds)

    # layerThickness = (ssh + H) broadcast over nVertLevels
    layer_thickness = ds['ssh'] + H
    layer_thickness, _ = xr.broadcast(layer_thickness, ds.refBottomDepth)
    ds['layerThickness'] = layer_thickness.transpose(
        'nCells', 'nVertLevels'
    ).expand_dims(dim='Time', axis=0)

    # Normal velocity initial condition: at rest
    norm_vel = xr.zeros_like(ds_mesh.xEdge)
    norm_vel, _ = xr.broadcast(norm_vel, ds.refBottomDepth)
    ds['normalVelocity'] = norm_vel.transpose(
        'nEdges', 'nVertLevels'
    ).expand_dims(dim='Time', axis=0)

    # Zonal wind stress forcing: tau_x = -tau_0 * cos(pi * (y - y_min) / Ly)
    ly_m = LY * 1e3
    wind_stress_zonal = -TAU_0 * np.cos(
        np.pi * (ds.yCell - ds.yCell.min()) / ly_m
    )
    ds['windStressZonal'] = wind_stress_zonal.expand_dims(dim='Time', axis=0)
    ds['windStressMeridional'] = xr.zeros_like(ds.xCell).expand_dims(
        dim='Time', axis=0
    )

    write_netcdf(ds, os.path.join(output_dir, 'initial_state.nc'))

    m = munk_layer_width()
    print(f'Done: {output_dir}/initial_state.nc')
    print(f'  resolution={resolution_km} km, nx={nx}, ny={ny}')
    print(f'  Munk layer width ~{m / 1e3:.1f} km '
          f'({m / dc:.1f} cells)')
    if m <= 3.0 * dc:
        print('  WARNING: resolution is too coarse to resolve the Munk layer '
              '(want >= 3 cells)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate barotropic gyre ICs for Moka.jl'
    )
    parser.add_argument('--res', type=float, required=True,
                        help='Grid resolution in km (e.g. 20, 10)')
    parser.add_argument('--dir', type=str, required=True,
                        help='Output directory (e.g. 10km)')
    args = parser.parse_args()

    create_initial_state(args.res, args.dir)
