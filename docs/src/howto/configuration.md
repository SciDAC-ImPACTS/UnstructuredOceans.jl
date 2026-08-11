# Configure a simulation

A MOKA run is fully specified by one YAML configuration file. The format follows
the OMEGA / MPAS-Ocean convention: a top-level `omega:` mapping that the parser
(`MOKA.config_read`, in `src/infra/Config.jl`) splits into
a **namelist** (grouped scalar options) and **streams** (I/O definitions).

Below is a complete config for the 10 km barotropic gyre
(`examples/barotropic_gyre/10km/config.yml`), annotated section by section.

## Namelist sections

```yaml
omega:
  run_modes:
    config_ocean_run_mode: forward
  time_management:
    config_do_restart: false
    config_start_time: 0001-01-01_00:00:00   # instant: YYYY-MM-DD_HH:MM:SS
    config_stop_time: none                    # set this OR run_duration
    config_run_duration: 0020_00:00:00        # interval: DDDD_HH:MM:SS (20 days)
    config_calendar_type: noleap
  io:
    config_write_output_on_startup: true
  time_integration:
    config_dt: 0000_00:00:40                  # 40 s timestep
    config_time_integrator: RungeKutta4       # or RK4 / ForwardEuler
    config_number_of_time_levels: 2
  forcing:
    config_use_bulk_wind_stress: true
  hmix_del2:
    config_use_mom_del2: true
    config_mom_del2: 400.0                     # Laplacian viscosity ν₂ [m² s⁻¹]
```

Key options:

- **Time span** — set either `config_stop_time` *or* `config_run_duration`
  (specifying both is checked for consistency). Timestamps use the MPAS formats
  `YYYY-MM-DD_HH:MM:SS` for instants and `DDDD_HH:MM:SS` for intervals. These are
  read into the [`Clock`](@ref) and its alarms.
- **Integrator** — `config_time_integrator` is mapped to a type by
  [`parse_integrator`](@ref): `RungeKutta4` (alias `RK4`) or `ForwardEuler`
  (aliases `Forward-Euler`, `euler`).
- **Timestep** — `config_dt` becomes the clock timestep; choose it for CFL
  stability at your resolution.
- **Viscosity** — `hmix_del2.config_mom_del2` sets ``\nu_2``; it is read by
  [`read_horz_mesh`](@ref) and stored on the mesh edges. Omit the section (or set
  `config_use_mom_del2: false`) to disable lateral mixing.
- **Wind forcing** — `forcing.config_use_bulk_wind_stress` (default `true`) gates
  the wind-stress term. The wind stress is read from the `streams.forcing` file
  (`windStressZonal`/`windStressMeridional`), projected onto edge normals, and
  stored in [`ForcingVars`](@ref) on the [`Mesh`](@ref); it is applied by the
  dispatched `WindForcing` term. Set the flag `false` to run without wind: the
  forcing is left zero, so the differentiated timestep path is unchanged.
- **Physical constants** — an optional `constants` section overrides the defaults
  built into the [`Constants`](@ref) carried on the [`Mesh`](@ref):
  `config_gravity` (``g``, default `9.80616`) and `config_density` (``\rho_0``,
  default `1000.0`). Both are stored as 1-element device arrays so kernels read
  them without breaking Enzyme AD. Omit the section to keep the defaults.

## Streams section

Streams declare the input, forcing, restart, and output files:

```yaml
  streams:
    mesh:
      filename_template: initial_state.nc
      input_interval: initial_only
    input:
      filename_template: initial_state.nc
      input_interval: initial_only
    forcing:
      filename_template: initial_state.nc
      contents: [windStressZonal, windStressMeridional]
    output:
      type: output
      filename_template: output.nc
      reference_time: 0001-01-01_00:00:00
      output_interval: 0030_00:00:00          # write every 30 days
      clobber_mode: truncate
      precision: double
      contents: [xtime, normalVelocity, layerThickness, ssh]
```

- **`mesh` / `input`** — where [`ocn_init`](@ref) reads the mesh geometry and the
  initial prognostic state.
- **`forcing`** — the wind-stress fields projected onto edge normals during mesh
  setup.
- **`output`** — `output_interval` sets the period of the output
  [`PeriodicAlarm`](@ref); `contents` lists the variables written each frame by
  [`io_write_timestep`](@ref).

## Generating meshes

The `initial_state.nc` / `culled_mesh.nc` inputs referenced above are produced by
the Python `setup.py` scripts in each example directory, which use the E3SM
[Polaris](https://docs.e3sm.org/polaris/) / MPAS-Tools toolchain. The committed
example directories already contain these files, so you only need Polaris if you
want to regenerate or add resolutions.
