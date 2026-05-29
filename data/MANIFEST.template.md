# VS-LeakKG data manifest (template)

Copy this file to `data/MANIFEST.md` (gitignored) and fill in the local
paths after you place the raw archives under `data/raw/`.

The repository contains **no datasets**. Everything below must be obtained
separately. Sizes are approximate.

## Required raw datasets

| Dataset | Source | Expected path | Size | Notes |
|---|---|---|---:|---|
| **ChEMBL 35 SQLite** | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz` | `data/raw/chembl/chembl_35_sqlite.tar.gz` → extracts to `data/raw/chembl/extracted/chembl_35/chembl_35_sqlite/chembl_35.db` | ~5 GB compressed / ~26 GB extracted | Needed for ChEMBL ligands, assays, documents, targets, component_sequences. |
| **BindingDB TSV** | `https://www.bindingdb.org/bind/downloads/BindingDB_All_<release>.tsv.zip` | `data/raw/bindingdb/BindingDB_All.tsv` | ~8 GB | Use any 2024+ release. |
| **PDBBind v2020 (general + refined)** | http://www.pdbbind.org.cn/ (registration required) | `data/raw/PBDBind/extracted/P-L/<year>/<pdbid>/{*_pocket.pdb, *_ligand.{mol2,sdf}, *_protein.pdb}` | ~19 GB | The directory layout is the PDBBind P-L file tree. |
| **LIT-PCBA full data** | `https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz` | `data/raw/LIT-PCBA/full_data/<TARGET>/{actives.smi, inactives.smi}` | ~600 MB | |
| **LIT-PCBA AVE-debiased splits** | `https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz` | `data/raw/LIT-PCBA/AVE_unbiased/<TARGET>/{actives_*.smi, inactives_*.smi}` | ~250 MB | Required for MVP-1 LIT-PCBA AVE audit. |
| **DUD-E** | `http://dude.docking.org/db/subsets/all/all.tar.gz` | `data/raw/DUD-E/all/<target>/{actives_final.ism, decoys_final.ism, receptor.pdb}` | ~100 MB | |
| **DEKOIS 2.0** | `https://pubmed.ncbi.nlm.nih.gov/23548029/` companion download | `data/raw/DEKOIS/<target>/{actives.sdf, decoys.sdf}` | ~500 MB | |
| **BayesBind V1.5** | https://figshare.com/articles/dataset/BayesBind_V1_5 | `data/raw/BayesBind/BayesBindV1.5/{val,test}/<TARGET>/{actives.csv, random.csv, pocket.pdb, rec*.pdb}` | ~220 MB | |
| **BigBind V1.5 metadata** | https://figshare.com/articles/dataset/BigBind | `data/raw/BigBind/BigBindV1.5/{activities_*,structures_*}.csv` | ~600 MB CSVs (~19 GB full archive optional) | Only the 12 top-level metadata CSVs are needed for the current audit; full structural archive is not consumed. |
| **PLINDER** (optional) | https://www.plinder.sh/ | not used in this pipeline | n/a | The full PLINDER download is forbidden per project rules; only metadata cross-checks would be considered. |

## Optional model checkpoints

| Model | Location | Path | Auth |
|---|---|---|---|
| ConGLUDe | bundled in repo `external/model_eval/conglude/checkpoints/best_model/` after cloning | ~13 MB | none |
| DrugCLIP | Google Drive `1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV` | `data/raw/model_checkpoints/drugclip/` | manual confirm |
| LigUnity | figshare DOI `10.6084/m9.figshare.27966819` | `data/raw/model_checkpoints/ligunity/` | manual confirm for >1 GB |
| HypSeek | author email | `data/raw/model_checkpoints/hypseek/` | author contact |

## How the pipeline uses these

1. `setup_data.sh` (under `scripts/`) downloads and extracts the public archives
   (ChEMBL, BindingDB, LIT-PCBA, DUD-E, DEKOIS, BayesBind, BigBind metadata).
2. Each load module under `src/vsleakkg/load_*.py` reads from the expected
   path above and writes a parquet under `data/processed/`.
3. Graph build (`src/vsleakkg/build_graph.py`, `run_overnight.py`) joins the
   per-dataset parquets and emits the heterogeneous graph parquets.
4. Audit pipeline (`run_mvp_audit.sh`, `run_mvp1_audit.sh`, etc.) produces the
   final reports + tables + figures under `outputs/`.

## Disk footprint summary

| Layer | Size |
|---|---:|
| raw archives (compressed) | ~15 GB |
| raw archives extracted | ~75 GB |
| processed parquets | ~5 GB |
| graph parquets | ~1 GB (subset of processed) |
| outputs (reports + tables + figures) | ~10 MB |

Plan for at least **100 GB free** during a full reproduction.

## Re-running on Linux

This repo was developed on Windows. The processing pipeline is mostly
polars + Python and portable, but two steps need attention:

- **MMseqs2** clustering uses the Cygwin-bundled Windows binary at
  `C:\Tools\mmseqs2\mmseqs\bin\mmseqs.exe`. On Linux, install via
  `conda install -c bioconda mmseqs2` and update the path constant in
  `src/vsleakkg/pdbbind_cluster_proteins.py` and
  `src/vsleakkg/pdbbind_chembl_target_match.py`.
- **Foldseek** (optional, for 3D pocket clustering) — install via
  `conda install -c bioconda foldseek`. The default pipeline runs the
  AA-composition MVP and does not require Foldseek.
