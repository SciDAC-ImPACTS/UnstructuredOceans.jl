```@meta
CurrentModule = MOKA
```

# API

The public API of MOKA, grouped by role. All symbols listed here are exported.

```@contents
Pages = ["api.md"]
Depth = 2
```

## Architectures

```@docs
CPU
GPU
AbstractArchitecture
AbstractSerialArchitecture
device
architecture
array_type
on_architecture
set_ad_device_heap!
DEFAULT_NTHREADS
```

## Mesh

```@docs
Mesh
HorzMesh
VerticalMesh
Constants
read_horz_mesh
Cell
Edge
Vertex
```

## Operators

```@docs
DivergenceOnCell!
GradientOnEdge!
CurlOnVertex!
ZeroOutVector!
```

## Model state

```@docs
PrognosticVars
DiagnosticVars
```

## Time management

```@docs
Clock
AbstractAlarm
OneTimeAlarm
PeriodicAlarm
advance!
is_ringing
change_time_step!
reset!
```

## Time integration

```@docs
RungeKutta4
ForwardEuler
parse_integrator
ocn_timestep
```

## Initialization

```@docs
ocn_init
ocn_init_alarms
ocn_init_shadows
```

## Run loop and automatic differentiation

```@docs
ocn_run_loop
ocn_run_loop_fwd!
ocn_run_loop_checkpointed!
ocn_loss
OceanModel
ocn_step!
```

## Input / output

```@docs
io_initialize
io_write_timestep
io_finalize
write_netcdf
```
