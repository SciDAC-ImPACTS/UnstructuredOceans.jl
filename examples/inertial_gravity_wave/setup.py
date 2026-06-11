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

# Physics parameters (from config.cfg defaults)
F_0       = 1e-4     # Coriolis parameter [s⁻¹]
G         = 9.80665  # standard gravity [m s⁻²]
ETA_0     = 1.0      # SSH amplitude [m]
H         = 1000.0   # resting depth [m]
LX        = 10000.0  # domain width [km]
N_WAVES_X = 2        # number of wavelengths in x
N_WAVES_Y = 2        # number of wavelengths in y


def _wavenumbers():
    ly = np.sqrt(3.0) / 2.0 * LX
    kx = N_WAVES_X * 2.0 * np.pi / (LX * 1e3)
    ky = N_WAVES_Y * 2.0 * np.pi / (ly * 1e3)
    omega = np.sqrt(F_0**2 + G * H * (kx**2 + ky**2))
    return kx, ky, omega


def exact_ssh(ds, t=0.0, eta0=ETA_0, f0=F_0, g=G, h=H):
    """Exact IGW SSH: η = η₀ cos(kx·x + ky·y − ω·t)."""
    kx, ky, omega = _wavenumbers()
    return eta0 * np.cos(kx * ds.xCell + ky * ds.yCell - omega * t)


def exact_normal_velocity(ds, t=0.0, eta0=ETA_0, f0=F_0, g=G, h=H):
    """Exact IGW normal velocity projected onto each edge normal."""
    kx, ky, omega = _wavenumbers()
    phase = kx * ds.xEdge + ky * ds.yEdge - omega * t
    amp = eta0 * g / (omega**2 - f0**2)
    u = amp * (omega * kx * np.cos(phase) - f0 * ky * np.sin(phase))
    v = amp * (omega * ky * np.cos(phase) + f0 * kx * np.sin(phase))
    return u * np.cos(ds.angleEdge) + v * np.sin(ds.angleEdge)


def create_initial_state(resolution_km, output_dir):
    """Create initial_state.nc for one resolution.

    Parameters
    ----------
    resolution_km : float
        Grid cell spacing in kilometres.
    output_dir : str
        Directory where output files are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    ly = np.sqrt(3.0) / 2.0 * LX
    dc = resolution_km * 1e3
    nx, ny = compute_planar_hex_nx_ny(LX, ly, resolution_km)

    # Periodic mesh (no land boundaries for IGW)
    ds_mesh = make_planar_hex_mesh(
        nx=nx, ny=ny, dc=dc, nonperiodic_x=False, nonperiodic_y=False
    )
    ds_mesh = cull(ds_mesh)
    ds_mesh = convert(ds_mesh)

    write_netcdf(ds_mesh, os.path.join(output_dir, 'culled_mesh.nc'))

    # Constant Coriolis on cells, edges, and vertices
    for loc in ['Cell', 'Edge', 'Vertex']:
        ds_mesh[f'f{loc}'] = F_0 * xr.ones_like(ds_mesh[f'x{loc}'])

    # Vertical grid from config.cfg (ocean.cfg sets min_vert_levels default)
    config = PolarisConfigParser()
    config.add_from_package('polaris.ocean', 'ocean.cfg')
    config.add_from_file('config.cfg')

    ds = ds_mesh.copy()
    ds['ssh'] = xr.zeros_like(ds_mesh.xCell)
    ds['bottomDepth'] = H * xr.ones_like(ds_mesh.xCell)

    init_vertical_coord(config, ds)

    # SSH initial condition
    ssh = exact_ssh(ds).expand_dims(dim='Time', axis=0)
    ds['ssh'] = ssh

    # layerThickness = ssh + H, broadcast over nVertLevels
    layer_thickness = ssh + H
    layer_thickness, _ = xr.broadcast(layer_thickness, ds.refBottomDepth)
    ds['layerThickness'] = layer_thickness.transpose(
        'Time', 'nCells', 'nVertLevels'
    )

    # Normal velocity initial condition
    norm_vel = exact_normal_velocity(ds)
    norm_vel, _ = xr.broadcast(norm_vel, ds.refBottomDepth)
    ds['normalVelocity'] = norm_vel.transpose(
        'nEdges', 'nVertLevels'
    ).expand_dims(dim='Time', axis=0)

    write_netcdf(ds, os.path.join(output_dir, 'initial_state.nc'))

    kx, ky, omega = _wavenumbers()
    print(f'Done: {output_dir}/initial_state.nc')
    print(f'  resolution={resolution_km} km, nx={nx}, ny={ny}')
    print(f'  kx={kx:.4e} m⁻¹, ky={ky:.4e} m⁻¹, ω={omega:.4e} s⁻¹')
    print(f'  period={2*np.pi/omega/3600:.2f} h')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate inertial gravity wave ICs for Moka.jl'
    )
    parser.add_argument('--res', type=float, required=True,
                        help='Grid resolution in km (e.g. 200, 100, 50, 25)')
    parser.add_argument('--dir', type=str, required=True,
                        help='Output directory (e.g. 200km)')
    args = parser.parse_args()

    create_initial_state(args.res, args.dir)
