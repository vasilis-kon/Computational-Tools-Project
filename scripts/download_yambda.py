"""Download the Yambda-50M data files from Hugging Face into this project.

Dataset: https://huggingface.co/datasets/yandex/yambda (Apache 2.0 license)

Files are saved in <project>/data/raw/yambda/, keeping the same folder
structure as on Hugging Face. Only the Python standard library is used,
so no extra packages are needed.

Run (works from any folder):
    python scripts/download_yambda.py                  # events + mappings (~477 MB)
    python scripts/download_yambda.py --embeddings     # also the audio embeddings (13.8 GB!)

The embeddings file covers all 7.72M tracks of the full 5B dataset, although
Yambda-50M only uses about 934k of them. It is optional for that reason.
An interrupted download is resumed when you run the script again.
"""

import argparse
import http.client
import time
import urllib.request
from pathlib import Path

# Fixed version (git commit) of the dataset on Hugging Face. Using a fixed
# version instead of "main" means everybody downloads exactly the same files,
# even if Yandex updates the dataset later.
REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
BASE_URL = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/"

# Files to download: path on Hugging Face -> expected size in bytes.
# The sizes come from the Hugging Face file listing. We use them to check
# that a download is complete.
FILES = {
    "flat/50m/multi_event.parquet": 384_465_869,
    "artist_item_mapping.parquet": 38_274_621,
    "album_item_mapping.parquet": 54_528_894,
}
EMBEDDINGS = {"embeddings.parquet": 13_814_230_943}

TIMEOUT_S = 30            # give up on a connection after this many seconds without data
MAX_ATTEMPTS = 200        # resume attempts per file
CHUNK = 1 << 20           # read 1 MB at a time
REPORT_EVERY = 250_000_000  # print progress every 250 MB

# Paths are relative to this script, so the data always ends up inside the
# project, no matter which folder the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "yambda"


def download_file(repo_path, expected_size):
    target = DATA_DIR / repo_path
    done = target.stat().st_size if target.exists() else 0

    # Skip files that were already downloaded completely.
    if done == expected_size:
        print(f"Already downloaded: {repo_path}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    # A stalled or dropped connection raises an error after TIMEOUT_S seconds
    # without data. We then resume from the bytes already on disk and try again.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        done = target.stat().st_size if target.exists() else 0
        if done == expected_size:
            break
        request = urllib.request.Request(BASE_URL + repo_path)
        if 0 < done < expected_size:
            request.add_header("Range", f"bytes={done}-")
            print(f"Resuming {repo_path} at {done / 1e6:.0f} of {expected_size / 1e6:.0f} MB "
                  f"(attempt {attempt}) ...", flush=True)
        else:
            done = 0
            print(f"Downloading {repo_path} ({expected_size / 1e6:.0f} MB) (attempt {attempt}) ...", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                # If the server ignored the Range header, it sends the whole file again.
                resumed = done > 0 and response.status == 206
                position = done if resumed else 0
                next_report = position + REPORT_EVERY
                with open(target, "ab" if resumed else "wb") as f:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        position += len(chunk)
                        if position >= next_report:
                            print(f"  {position / 1e6:,.0f} / {expected_size / 1e6:,.0f} MB "
                                  f"({100 * position / expected_size:.0f}%)", flush=True)
                            next_report += REPORT_EVERY
        except (OSError, http.client.HTTPException) as error:   # includes timeouts and dropped connections
            print(f"  connection problem: {error!r}; retrying in 5 s", flush=True)
            time.sleep(5)

    # A wrong size means the download could not be completed.
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{repo_path}: expected {expected_size} bytes, got {actual_size}. "
            "Please run the script again."
        )
    print(f"  saved to {target.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Download the Yambda-50M files.")
    parser.add_argument("--embeddings", action="store_true",
                        help="also download embeddings.parquet (13.8 GB)")
    args = parser.parse_args()

    files = dict(FILES)
    if args.embeddings:
        files.update(EMBEDDINGS)

    for repo_path, expected_size in files.items():
        download_file(repo_path, expected_size)
    print("Done. All files are in", DATA_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
