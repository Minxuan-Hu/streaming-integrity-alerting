# Streaming Integrity Alerting

Code and archived experiment outputs for **A Deployment-Oriented Framework for Evaluating Streaming Integrity Alerting Under Workload Constraints**.

This repository contains the code and archived outputs supporting the paper and does not redistribute the source datasets or constructed data panels.

The repository supports two workflows:

- `python reproduce.py` regenerates the manuscript tables and Figures 2-4 from the archived experiment outputs.
- `scripts/run_evaluation.py` reruns an experiment after the required data panel has been constructed locally.

Source and processed datasets are not redistributed.

## Reproduce the paper outputs

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

A successful run writes:

```text
Reproduction completed successfully.
Tables:  results/tables
Figures: results/figures
Checks:  results/validation.csv
```

The final status is also recorded in `results/reproduction_summary.json`.

This quick reproduction uses the compact archived experiment outputs and does not require downloading the source datasets. Full experiment reruns require locally constructed panels and are described below.

## Artifact checklist

A complete quick reproduction verifies the following repository objects:

- environment and source-data instructions in `README.md` and `requirements.txt`;
- deterministic CFPB and SMAP panel builders;
- compact outputs and exact manifests for 16 saved experiment configurations in `data/runs/`;
- fixed detector, trial, and bootstrap-resampling manifests in `data/manifests/`;
- analysis scripts that regenerate all 14 manuscript result tables and Figures 2-4;
- the 343-row online/replay parity audit at `data/runs/cfpb_online_recal1_seed0/dev_invariants_pstar_parity.csv`;
- reference checksums in `expected/`; and
- 239 automated numerical checks recorded in `results/validation.csv` and summarized in `results/reproduction_summary.json`.

The expected final validation status is 239 passed and zero failed checks.

## Source data and panel construction

Download the source datasets from their original providers:

- [CFPB Consumer Credit Trends](https://www.consumerfinance.gov/data-research/consumer-credit-trends/): download the full data CSV.
- [NASA SMAP data in Telemanom format](https://github.com/khundman/telemanom): obtain the archive containing `data/train/*.npy` and `data/test/*.npy`.

Place the downloaded files anywhere outside the repository, then construct the panels:

```bash
python build_cfpb_panel.py \
  --raw_csv /path/to/all_data.csv \
  --out_csv data/panels/cfpb_panel.csv \
  --out_pkl data/panels/cfpb_panel.pkl \
  --out_meta data/panels/cfpb_signals_metadata.json

python make_smap_panel.py \
  --smap_zip /path/to/SMAP.zip \
  --series_id A-1 \
  --out_dir data/panels \
  --downsample_k 7 \
  --downsample_mode mean \
  --n_score_cols 10

python make_smap_panel.py \
  --smap_zip /path/to/SMAP.zip \
  --series_id D-15 \
  --out_dir data/panels \
  --downsample_k 7 \
  --downsample_mode mean \
  --n_score_cols 10
```

The A-1 experiments use the source-preserving split derived from the separately downsampled Telemanom arrays: training ends on the synthetic date `2007-11-10`, and evaluation begins on `2007-11-17`. The D-15 split is `2005-08-27` / `2005-09-03`. SMAP dates are synthetic and encode ordering and weekly cadence only.

## Rerun an experiment

After constructing the panels, list the saved configurations and rerun one by name:

```bash
python scripts/run_evaluation.py --list
python scripts/run_evaluation.py smap_A1_weekly_ledger_seed0
```

Full reruns are written under `results/full_evaluations/`. They do not overwrite the archived outputs used by `python reproduce.py`.

Full reruns are substantially more computationally expensive than the quick reproduction. Each saved configuration has an exact manifest in its corresponding `data/runs/<run-name>/run_manifest.json` file.

## Repository layout

- Root Python modules: evaluation framework, detectors, incidents, standardization, and panel adapters
- `analysis/`: paper-level summaries, uncertainty calculations, table generation, figure generation, and validation
- `data/runs/`: compact archived experiment outputs used by `reproduce.py`
- `data/manifests/`: fixed detector menus, aligned trials, and bootstrap resampling indices
- `expected/`: reference checksums used by validation
- `results/`: regenerated paper tables, figures, and validation output
- `scripts/run_evaluation.py`: saved-configuration rerun helper

## License

The repository code is licensed under the Apache License 2.0. Source datasets are governed by their respective providers and are not included.
