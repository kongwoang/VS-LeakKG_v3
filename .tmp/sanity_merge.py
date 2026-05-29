"""Sanity-check the v3 KG node merging."""
import polars as pl
from pathlib import Path
P = Path("/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v3/data/processed")
nodes = pl.read_parquet(P / "mvp2_nodes.parquet")
edges = pl.read_parquet(P / "mvp2_edges.parquet")

print("=" * 60)
print("CHECK 1: Duplicate node_id rows (should be 0)")
print("=" * 60)
dup = nodes.group_by("node_id").len().filter(pl.col("len") > 1)
print(f"  duplicate node_id: {dup.height}")
if dup.height:
    print(dup.head(5))

print()
print("=" * 60)
print("CHECK 2: Ligands shared across multiple benchmark corpora")
print("=" * 60)
# Find ligand IDs touched by example_has_ligand edges
ehl = edges.filter(pl.col("edge_type") == "example_has_ligand").select(["src", "dst"])
ex_src_map = nodes.filter(pl.col("node_type") == "Example").select(
    pl.col("node_id").alias("src"),
    pl.col("label").alias("example_label"),
)
# Examples typically have node_id like "ex:litpcba_ave:42" or "ex:dude:99"
ehl2 = ehl.join(ex_src_map, on="src", how="left").with_columns(
    pl.col("src").str.split(":").list.get(1).alias("corpus")
)
lig_corpus = ehl2.group_by("dst").agg(pl.col("corpus").n_unique().alias("n_corp"),
                                       pl.col("corpus").unique().alias("corpora"))
multi_corp = lig_corpus.filter(pl.col("n_corp") >= 2)
print(f"  total ligands with example_has_ligand: {lig_corpus.height:,}")
print(f"  ligands in >=2 corpora (real leak signal): {multi_corp.height:,}")
print("  top 5 corpora-combination cohorts:")
print(multi_corp.group_by("n_corp").len().sort("n_corp", descending=True).head(5))

print()
print("=" * 60)
print("CHECK 3: same_inchikey_as edges (cross-source ligand linking)")
print("=" * 60)
sik = edges.filter(pl.col("edge_type") == "same_inchikey_as")
print(f"  same_inchikey_as edges: {sik.height:,}")
print("  Sample:")
print(sik.head(3))

print()
print("=" * 60)
print("CHECK 4: ligand_exact edges (v2 InChIKey-level merge)")
print("=" * 60)
lex = edges.filter(pl.col("edge_type") == "ligand_exact")
print(f"  ligand_exact edges: {lex.height:,}")
print(lex.head(3))

print()
print("=" * 60)
print("CHECK 5: Example nodes per source (verify no Example merge)")
print("=" * 60)
ex_nodes = nodes.filter(pl.col("node_type") == "Example")
ex_nodes2 = ex_nodes.with_columns(
    pl.col("node_id").str.split(":").list.get(1).alias("corpus")
)
print(ex_nodes2.group_by("corpus").len().sort("len", descending=True))
