# U_Net_3D workspace layout

The subproject resolves every runtime path through `project_paths.py`, which
uses `utils.paths.SubprojectPaths`. The repository contains source code and
documentation only.

## Roots

- Raw, read-only data: `<HYDRO_DATA_DIR>`
- Reusable derived data: `<HYDRO_WORKSPACE>/cache/U_Net_3D`
- Experiment products: `<HYDRO_WORKSPACE>/results/U_Net_3D/<experiment>`

`HYDRO_WORKSPACE` and `HYDRO_DATA_DIR` are configured in the repository-root
`.env`. Dataset-specific drive paths are not embedded in Python source.

## Managed data

- `cache/U_Net_3D/prepared`: reusable arrays produced by preprocessing
- `cache/U_Net_3D/station_observations`: station table and rebuilt observation grids
- `results/U_Net_3D/<experiment>/checkpoints`: model weights
- `results/U_Net_3D/<experiment>/figures`: figures
- `results/U_Net_3D/<experiment>/logs`: logs
- `results/U_Net_3D/artifact_archive`: results generated before this refactor

## Migration map

- Old `U_Net_3D/*.npz`, metrics, and logs -> `artifact_archive`
- Old external `*_run` directories -> `results/U_Net_3D/<original_name>`
- Old `lr_unet`, `hr_unet`, and `sst_6chan.npy` -> `cache/U_Net_3D/prepared`
- Rebuilt station observations and `observe_data24.csv` ->
  `cache/U_Net_3D/station_observations`
- Old root-level multi-lead arrays -> `results/U_Net_3D/legacy_default`
- Old weights, figures, and logs -> `results/U_Net_3D/legacy_unet`

Command-line path arguments remain available for reproducibility. When an
output argument is omitted, scripts use their managed experiment directory.
