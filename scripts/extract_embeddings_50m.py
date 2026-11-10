"""Extract the audio embeddings of the tracks that occur in Yambda-50M.

embeddings.parquet (13.8 GB) holds 7.72M tracks as double-precision lists, but
Yambda-50M only uses about 0.9M of them. This script keeps those tracks and
stores them compactly (float32) so notebooks can load them in seconds:

    data/processed/emb50m_item_ids.npy   (n,)      uint32  track ids
    data/processed/emb50m_embed.npy      (n, d)    float32 raw embeddings
    data/processed/emb50m_norm.npy       (n, d)    float32 normalised embeddings

Run (works from any folder):
    python scripts/extract_embeddings_50m.py
Memory: about 1-2 GB (one row group at a time). Runtime: a few minutes.
"""

import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "yambda"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
# tracks that occur in Yambda-50M (any event type)
items = pq.read_table(DATA / "flat/50m/multi_event.parquet", columns=["item_id"]).column("item_id")
wanted = pc.unique(items)
print(f"{len(wanted):,} distinct tracks in Yambda-50M")

pf = pq.ParquetFile(DATA / "embeddings.parquet")
ids, emb, norm = [], [], []
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g)
    t = t.filter(pc.is_in(t.column("item_id"), value_set=wanted))
    if t.num_rows:
        ids.append(t.column("item_id").to_numpy())
        for name, store in (("embed", emb), ("normalized_embed", norm)):
            col = t.column(name).combine_chunks()
            d = len(col[0])
            store.append(col.values.to_numpy().astype(np.float32).reshape(-1, d))
    print(f"  row group {g + 1}/{pf.metadata.num_row_groups}: kept {sum(len(i) for i in ids):,} tracks "
          f"({time.time() - t0:.0f} s)", flush=True)

ids = np.concatenate(ids); emb = np.concatenate(emb); norm = np.concatenate(norm)
order = np.argsort(ids)
np.save(OUT / "emb50m_item_ids.npy", ids[order])
np.save(OUT / "emb50m_embed.npy", emb[order])
np.save(OUT / "emb50m_norm.npy", norm[order])
print(f"saved {len(ids):,} tracks, dimension {emb.shape[1]} -> {OUT.relative_to(ROOT)}  "
      f"({100 * len(ids) / len(wanted):.1f}% of the tracks in Yambda-50M have an embedding)")
