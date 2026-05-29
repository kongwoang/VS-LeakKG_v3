# VS-LeakKG v3

Clean starting point for the next iteration of the VS-LeakKG benchmark-integrity framework. Inherits only the **KG-construction core** from v1; downstream audit, scoring, split, and baseline code is intentionally omitted and will be redesigned in this repo.

See `MOVE_REPORT.md` for the exact list of files carried over from `D:\hoangpc\VS-LeakKG` and what was deliberately left behind.

## What's here

```
src/vsleakkg/
  graph_schema.py       node/edge type enums + prefixed-ID conventions
  build_graph.py        per-corpus nodes/edges emitter
  chem.py               RDKit canonicalisation, Bemis-Murcko scaffolds, ECFP4
  io.py                 IO helpers
  load_*.py             dataset loaders (LIT-PCBA, DUD-E, DEKOIS, PDBBind, ChEMBL, BindingDB, BayesBind)
  run_pdbbind.py        PDBBind-specific KG construction (Complex/Pocket/StructureFile emission)
  run_overnight.py      ChEMBL/BindingDB mvp2 KG construction (also contains downstream audit tasks — slated for removal)

scripts/
  _dataset_version.ps1 / dataset_version.sh   single source of truth for HF dataset archive version
  fetch_dataset.{ps1,sh}                       pull dataset archive from kongwoang/VS_LeakKG on Hugging Face
  extract_datasets.{ps1,sh}                    extract pulled archive into data/raw + data/processed
  download_full_cache.{ps1,sh}                 alt path: cache the full HF repo
  setup_data.sh                                end-to-end data bootstrap
```

## Quick start

```bash
# 1. Set up Python env (RDKit, polars, mmseqs2 optional)
conda env create -f environment.yml
conda activate vsleakkg

# 2. Pull datasets from Hugging Face (private repo — needs HF token)
bash scripts/fetch_dataset.sh
bash scripts/extract_datasets.sh
```

## Workflow: Windows <-> VUW

Code lives on Windows and is mirrored to VUW via git:

```powershell
# Windows
git push origin main

# VUW
git pull origin main      # or: git clone <repo> on first time
```

Remote: `https://github.com/kongwoang/VS-LeakKG_v3`.

## Status

Stub. Schema, graph builder, and dataset loaders are inherited verbatim from v1; everything downstream (contamination scoring, split generation, baselines, audit tools, paper figures) needs to be rewritten in this repo.
