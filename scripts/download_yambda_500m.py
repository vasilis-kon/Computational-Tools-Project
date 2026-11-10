"""Download the Yambda-500M event file from Hugging Face into this project.

Dataset: https://huggingface.co/datasets/yandex/yambda (Apache 2.0 license)

Downloads only what the scaling experiments need:
    flat/500m/multi_event.parquet   (3.84 GB, all event types of 100,000 users)

The artist/album mappings and the audio embeddings are shared by all dataset
sizes and are already downloaded by download_yambda.py (--embeddings).

Same fixed dataset version, folder layout and safety features as
download_yambda.py (timeout, automatic retry, resume, size check, progress).

Run (works from any folder):
    python scripts/download_yambda_500m.py            # download
    python scripts/download_yambda_500m.py --check    # only compare the size on the server, download nothing
"""

import argparse
import urllib.request

# reuse the download logic (resume, retries, progress) and the fixed dataset version
from download_yambda import BASE_URL, DATA_DIR, PROJECT_ROOT, TIMEOUT_S, download_file

FILES = {
    "flat/500m/multi_event.parquet": 3_843_474_075,
}


def check_remote_sizes():
    for repo_path, expected in FILES.items():
        request = urllib.request.Request(BASE_URL + repo_path, method="HEAD")
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            remote = int(response.headers["Content-Length"])
        status = "OK" if remote == expected else "MISMATCH"
        print(f"{repo_path}: server {remote:,} bytes, expected {expected:,} -> {status}")


def main():
    parser = argparse.ArgumentParser(description="Download the Yambda-500M event file.")
    parser.add_argument("--check", action="store_true", help="only check the file size on the server")
    args = parser.parse_args()

    if args.check:
        check_remote_sizes()
        return

    for repo_path, expected_size in FILES.items():
        download_file(repo_path, expected_size)
    print("Done. All files are in", DATA_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
