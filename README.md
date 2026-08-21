# emsurr: learned EM surrogate baseline for PCB interconnects

Research question: can a learned model predict high-frequency S-parameters
for PCB structures it has never seen, and detect when it is too uncertain
and should fall back to a full-wave solver?

This repo is the first milestone: dataset pipeline, leakage-checked
evaluation splits (IID vs held-out topology families vs parameter
extrapolation), three baselines, an uncertainty/solver-fallback study, and
an openEMS smoke test.

## Data

The target dataset is the [TU Hamburg SI/PI-Database](https://www.tet.tuhh.de/en/si-pi-database/)
(Schierholz et al., IEEE Access 2021). Access is form-gated and the license
forbids redistribution, so **no TUHH data is in this repo**; see
`docs/data_audit.md` for the audit, license terms, and the manual access
procedure. Until access is granted, a clearly-labeled synthetic
physics-based dataset (7 topology families of cascaded transmission-line
elements, `src/emsurr/synth.py`) exercises the identical pipeline.
`src/emsurr/tuhh.py` loads the real archives once they are placed under
`data/raw/tuhh/`.

## Layout

```
src/emsurr/     physics.py synth.py dataset.py tuhh.py splits.py
                models.py metrics.py uncertainty.py
configs/        baseline.yaml
scripts/        generate_data.py run_baselines.py openems_smoke.py
tests/          test_pipeline.py
docs/           data_audit.md evaluation_protocol.md
results/        metrics.json baseline_report.md plots
```

## Reproduce

```bash
pip install -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests   # sanity
python scripts/generate_data.py                            # build dataset + checks
python scripts/run_baselines.py                            # all splits, models, uncertainty
```

Models are tiny and train on CPU by design (the GPU is shared; submit any
future GPU jobs via `pcslurm submit`).

## Findings

See `results/baseline_report.md`.

## Citation obligations

Any use of TUHH data must cite: M. Schierholz et al., "SI/PI-Database of
PCB-Based Interconnects for Machine Learning Applications," IEEE Access,
vol. 9, pp. 34423-34432, 2021, doi:10.1109/ACCESS.2021.3061788.
