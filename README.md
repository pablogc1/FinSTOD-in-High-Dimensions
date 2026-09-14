# FinSTOD in high dimensions

Python library and production pipeline for **FinSTOD** (Finite-time Strong
Trajectorial Ontological Differentiation) on \(m\)-dimensional flows,
\(m = 2\)–\(32\).

This repository accompanies the high-dimensional study. The two-dimensional
method and its first applications are in

> P. García-Cuadrillero, J. A. Capitán, F. Revuelta,
> *Finite-time Strong Trajectorial Ontological Differentiation (FinSTOD): a
> trajectory-based indicator of Lagrangian coherent structures*,
> Commun. Nonlinear Sci. Numer. Simul. **163**, 110763 (2026).
> [doi:10.1016/j.cnsns.2026.110763](https://doi.org/10.1016/j.cnsns.2026.110763)

FinSTOD compares neighbouring discrete trajectories by cancellation of visited
cells. It does not use the variational equations. Production fields in the
paper are \(1000\times1000\) sections (two coordinates; the rest held at a
reference value), compared with variational FTLE on the same cells, and with
FLI and arc-length Lagrangian descriptors (\(p=1\)) where the pictures still
differ. Median termination depth \(k^{*}/L\) is the built-in validity check:
if the comparison does not finish inside the window, the score stops reporting
the flow.

The method is **not** a cheaper drop-in FTLE. On the timed suite it is slower
in fourteen of fifteen comparisons; the two codes do different work per cell.

## Layout

```
stod_nd/                 FinSTOD pair logic, flows, FTLE / FLI / LD
tests/                   correctness tests (including the 2-D published rule)
cesvima_pipeline/
  pipeline_core/         shard, run, aggregate
  configs/               YAML for every production and sweep run
  master_pipeline.slurm  Slurm entry point
  worker.slurm           FinSTOD + FTLE array workers
  worker_aux.slurm       LD / FLI / time-aware FTLE on existing tags
diagnostics/             paper figure scripts (optional)
docs/                    interactive ABC and Lorenz-96 volume cubes
requirements.txt
```

Production NPZ fields are not in this repository (they are large). The YAML
configs are the complete recipe.

## Install

Python 3.9+ with NumPy and Matplotlib. The pipeline also needs PyYAML.

```bash
pip install numpy matplotlib pyyaml
python tests/test_core.py
```

`pytest tests/` works if you have pytest. There is no compiled backend.

## Library

```python
from stod_nd import Lorenz96, slice_cells, evaluate_cells, flow_ftle, spearman

sys = Lorenz96(m=4, F=8.0, dt_out=0.002)
cells, ticks = slice_cells(m=sys.m, n_cells=80, axes=(0, 1), stride=4)
out = evaluate_cells(sys, cells, n_cells=80, n_levels=200)
ftle = flow_ftle(sys, cells, n_cells=80, n_levels=200)
print(spearman(out["raw_scores"], ftle))
```

Flows with analytic Jacobians: Lorenz-96, ABC, Froeschlé, Hénon–Heiles, FPUT,
Kuramoto, CR3BP, coupled Duffing, Bickley jet, integrable torus. Neighbourhood
evaluation uses \(2m+1\) trajectories per cell (the seed and its von Neumann
neighbours).

## Local pipeline smoke test

From the repository root (not from `cesvima_pipeline/`):

```bash
python cesvima_pipeline/pipeline_core/generate_tasks.py \
    --config cesvima_pipeline/configs/local_quick_check.yaml

python cesvima_pipeline/pipeline_core/run_worker.py \
    --run_dir cesvima_output/local_quick_test --task_id 0
# repeat --task_id 1 2 3

python cesvima_pipeline/pipeline_core/aggregate_results.py \
    --run_dir cesvima_output/local_quick_test
```

That config is a \(20\times20\) Lorenz-96 slice, four shards, \(L=200\). It
should finish in a few seconds per shard. Production configs use
`n_cells: 1000` and `num_shards` up to 75; those are cluster jobs.

## Reproducing the paper runs

Each row of Supplementary Table S1 corresponds to a YAML file in
`cesvima_pipeline/configs/`. Naming:

| Prefix | Role |
| :--- | :--- |
| `prod_*` | production \(1000\times1000\) sections |
| `prod_*_3d_volume` | volumetric cubes (ABC \(120^3\); Hénon–Heiles and Lorenz-96 \(100^3\)) |
| `interp_*` | extra Duffing / Bickley windows and slices |
| `test2_bickley_*` | Bickley jet, pure shear, and one-cell origin shifts |
| `aux_interp.yaml` | LD (\(p=1\) and \(p=0.5\)), FLI, FTRN, time-aware FTLE on existing tags |
| `aux_fput_sbar.yaml` | FPUT modal \(\bar S\) |
| `prod_hh_energy_*_v2.yaml` | Hénon–Heiles energy sweep (**use `_v2` only**) |
| `toy_*`, `pilot_*`, `local_quick_check.yaml` | tiny jobs for debugging |

On a Slurm cluster, from `cesvima_pipeline/`:

```bash
sbatch --export=ALL,CFG=configs/prod_froeschle_dof2_m4.yaml master_pipeline.slurm
```

`sbatch master_pipeline.slurm` runs the full phased campaign. Edit partition
names, modules, and wall times in the `.slurm` files for your site; they are
written for Magerit (Universidad Politécnica de Madrid). Array width is capped
at 75 shards per run.

Paper figures from aggregated NPZs:

```bash
python diagnostics/plot_paper_atlas.py
python diagnostics/plot_paper_indicators.py
```

## Interactive cubes

The ABC and Lorenz-96 volumes are rotatable cubes (field on the six faces).
GitHub’s file view will not run them; they are served as GitHub Pages:

- <https://pablogc1.github.io/FinSTOD-in-High-Dimensions/>
- <https://pablogc1.github.io/FinSTOD-in-High-Dimensions/abc_volume_cube.html>
- <https://pablogc1.github.io/FinSTOD-in-High-Dimensions/lorenz_volume_cube.html>

Enable Pages once: repository **Settings → Pages → Build and deployment**,
source **Deploy from a branch**, branch **`main`**, folder **`/docs`**.
The Lorenz cube is a 3-D cut of an \(m=6\) system, not the full six-dimensional
state.

## Citation

Please cite the two-dimensional paper above and this repository:

```bibtex
@misc{FinSTOD_highdim_repo,
  author       = {Garc{\'\i}a-Cuadrillero, P.},
  title        = {{FinSTOD} in high dimensions},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/pablogc1/FinSTOD-in-High-Dimensions}}
}
```
