"""Quick look at the downloaded Yambda files (no analysis).

For each file it prints the number of rows, the columns with their types,
and the first 10 rows. Only those 10 rows are loaded into memory.

Run from the project root:
    uv run python scripts/peek_data.py
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 200)        # print all columns on one line
pd.set_option("display.max_columns", None)

# Paths are relative to this script, so it works from any start folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "yambda"

FILES = [
    DATA_DIR / "flat/50m/multi_event.parquet",
    DATA_DIR / "artist_item_mapping.parquet",
    DATA_DIR / "album_item_mapping.parquet",
]

for path in FILES:
    pf = pq.ParquetFile(str(path))              # opens the file, reads only its description
    print("=" * 80)
    print(path)
    print("rows:", pf.metadata.num_rows)     # number of rows, without loading the data
    print(pf.schema_arrow)                   # column names and types
    first_rows = next(pf.iter_batches(batch_size=10)).to_pandas()
    print(first_rows)                        # only the first 10 rows
