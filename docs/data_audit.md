# Data audit: TU Hamburg SI/PI-Database

Audited 2026-08-20 from the public pages of the Institut für Theoretische
Elektrotechnik (TET), TUHH: https://www.tet.tuhh.de/en/si-pi-database/

## Summary verdict

The SI/PI-Database is an excellent fit for this project's research question:
it contains ~79k parameter variations across 22 named PCB interconnect
structure families, each with complex S-parameters in Touchstone format and a
per-family `parameter.csv` describing every varied geometry/material
parameter. Topology-held-out (OOD) evaluation is genuinely possible because
families are distinct structures (different layer counts, via array sizes,
link vs array, SI vs PI regimes).

**However: the data is NOT programmatically downloadable.** Access requires a
manual request form (name, institution, position, email, structure
selection), acceptance of a license agreement, and a download link delivered
by email. Automated scraping of the data would violate the terms. Requesting
access is therefore a **manual action for the project owner** and is left as
such. Until the download arrives, this repo uses a clearly-labeled synthetic
physics-based stand-in dataset (see "Interim dataset" below) with the loader
designed around the TUHH file layout so real data slots in unchanged.

## Access procedure (manual)

1. Fill the contact form at https://www.tet.tuhh.de/en/si-pi-database/
   (name, institution, position, email, which structures you want).
2. Agree to the terms (license PDF linked on the page).
3. Receive a download link by email; place the extracted archives under
   `data/raw/tuhh/<dataset-id>/`.

## License / terms (from the license agreement, TET TUHH, March 2021)

- Free of charge for non-commercial, strictly non-armaments research use.
- **No redistribution**: may not distribute, sell, share, or make any part of
  the data available to third parties; use limited to the same
  department/group; hosting as a web service is explicitly forbidden.
  Consequence for this repo: raw and processed TUHH data must stay out of
  git and out of any published artifact. `.gitignore` excludes `data/`.
- No warranty; users must validate the data themselves.
- Users agree to be listed by institution on the database's user list.
- **Mandatory citation** whenever the data is used:
  M. Schierholz, A. Sanchez-Masis, A. Carmona-Cruz, X. Duan, K. Roy,
  C. Yang, R. Rimolo-Donadio, C. Schuster, "SI/PI-Database of PCB-Based
  Interconnects for Machine Learning Applications," IEEE Access, vol. 9,
  pp. 34423-34432, 2021, doi:10.1109/ACCESS.2021.3061788.
  Some datasets (e.g. SI-5..SI-7) additionally require citing
  T. Hillebrecht et al., "Generation And Application of a Very Large Dataset
  for Signal Integrity Via Array and Link Analysis" (submitted).

## Available structure families

Signal integrity (SI) datasets:

| ID | Structure | NSam | fmax | Freq pts | Cavities | Ports | Size |
|----|-----------|------|------|----------|----------|-------|------|
| SI-1 | Link on 11-cavity PCB, 10x10 via arrays | 7031 | 100 GHz | 199 | 11 | 12 | 2.4 GB |
| SI-2 | Link on 8-cavity PCB, 10x10 via arrays | 7031 | 100 GHz | 199 | 8 | 12 | 2.5 GB |
| SI-3 | 5x5 via array on 10-cavity PCB | 5000 | 40 GHz | 199 | 10 | 34 | 12.2 GB |
| SI-4 | Link on 10-cavity PCB, two 5x5 via arrays | 1500 | 40 GHz | 334 | 10 | 68 | 23.7 GB |
| SI-5 | Universal single-ended SI via array (LHS) | 1932 | 100 GHz | 400 | 3-47 | 6-58 | 7.5 GB |
| SI-6 | Universal differential SI via array (LHS) | 1916 | 100 GHz | 400 | 3-47 | 4-56 | 6.7 GB |
| SI-7 | Universal differential SI via array link (LHS) | 1073 | 100 GHz | 400 | 11-47 | 8-112 | 6.8 GB |
| SI-8 | 14-layer PCB, 9x8 via array (full wave) | 39 | 200 GHz | 400 | 13 | 44 | 390 MB |
| SI-9 | Coaxial via variation (full wave) | 72 | 200 GHz | 400 | 1-8 | 4 | 3 MB |

Power integrity (PI) datasets: PI-1..PI-13, all impedance/PDN structures to
1 GHz, 121-334 frequency points, 2-68 ports, 500-36200 samples each
(e.g. PI-1: PWR/GND plane pair with 11x11 via array, 36200 samples;
PI-2/PI-3: 4/8-layer PDN with two via arrays, 10000 samples each;
PI-10..13: 4-layer Eurocard PDN at increasing via-array complexity, 2000 each).

## Data format (per family, from the SI-5 structure document)

```
<family>.zip
├── <structure>.pdf      # human-readable structure description
├── parameter.csv        # one row per simulation; columns = varied params,
│                        # plus SIMULATION index linking to touchstone file
└── variation/           # Touchstone .sNp files, complex S-parameters
```

Varied parameters (SI-5 example): VIAS_X_AMOUNT, VIAS_Y_AMOUNT,
SIGNAL_AMOUNT, GROUND_AMOUNT, POWER_AMOUNT, PITCH, VIARADIUS,
ANTIPAD_RADIUS, LAYER_AMOUNT, TMET, TDIEL, CONDUCTIVITY, PERMITTIVITY,
LOSSTANGENT. Stackup is described per-simulation (`stackup.txt`), always
solid planes with two central PWR planes flanked by GND, PML lateral
boundaries. Frequency grid for SI-5: 250 MHz to 100 GHz, 400 linear points.

Simulator: TUHH physics-based via/plane model (CONMLS), cross-validated
against full-wave solvers; SI-8/SI-9 are full-wave references.

## Is topology-OOD evaluation possible?

Yes, at two levels:

1. **Across families**: hold out entire dataset IDs (e.g. train on SI-1..SI-3,
   test on SI-4). Caveats: frequency grids differ across families (199 vs 334
   vs 400 points, fmax 40 vs 100 vs 200 GHz), port counts differ, and SI vs
   PI are different physical regimes. A common-grid resampling policy and a
   port-subset policy are required; documented in
   `docs/evaluation_protocol.md`.
2. **Within universal families (SI-5..SI-7)**: layer count and array size
   vary per sample, so structural extrapolation splits (e.g. train on <=8
   layers, test on >8) exist even inside one family.

Parameter-extrapolation splits (unseen permittivity/loss tangent ranges,
geometry outside training support) are supported by every family since
`parameter.csv` exposes the full design vector.

Known limitation: most SI families are variations of the same physical class
(via arrays in plane cavities). "OOD topology" here means unseen structural
configuration, not an arbitrary new circuit class; the report must state
this.

## Interim dataset (until TUHH access is granted)

`src/emsurr/synth.py` generates a synthetic multi-family dataset from
analytic transmission-line physics (scikit-rf): microstrip/stripline lines,
coupled-line sections, open/short stubs, stepped-impedance sections, and
lumped via transitions, cascaded into 2-port networks with varied geometry,
dielectric, and stackup parameters. This is a *stand-in*: it validates the
pipeline, split machinery, baselines, metrics, and uncertainty protocol on
data with the same schema. All results on it are labeled `synthetic` and no
scientific claim about the TUHH data is made from it.
