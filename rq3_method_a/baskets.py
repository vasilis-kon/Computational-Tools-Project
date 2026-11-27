"""Step 2 of RQ3 Method A: turn the Yambda-50M listening log into session baskets of tracks.

Rules (the same as Viktor's Method B, so that both methods work on the same baskets; D-numbers refer to the plan):
  D1 real listen   : played_ratio_pct >= 50
  D2 stream        : scripts/streams.py, first-discovery rule with a 60-day warm-up (label -1 = excluded)
  D3 session       : a new session starts after a gap of more than 30 minutes
  D8 embedding     : keep only tracks that have an audio embedding (data/processed/emb50m_item_ids.npy);
                     --tracks all skips this filter, for as long as that file is not available
  D5 users         : at least 100 distinct real tracks in BOTH streams
  D4 basket        : the distinct tracks of one user, in one session, in one stream; the first 15 in time; at least 2
  D6 per-user cap  : at most 100 random baskets per user and stream
  D7 matching      : the same number of baskets of every size in both streams

Run from the project root (needs numpy, pandas, pyarrow; about 10 seconds; peak memory about 8 GB):
    python rq3_method_a/baskets.py                    # default: first-discovery rule, embedded tracks, seed 42
    python rq3_method_a/baskets.py --tracks all       # without the embedding filter D8 (no id list needed)
    python rq3_method_a/baskets.py --seed 7           # robustness: other random choices in D6 and D7
    python rq3_method_a/baskets.py --mode listen      # robustness: the old per-listen stream rule

Output (the settings are in the file names):
    data/processed/rq3_baskets_50m_<mode>_<tracks>_seed<seed>.parquet   one row per track in a basket:
                                                                        basket_id, uid, stream, size, position, item_id
    outputs/rq3_baskets_funnel_50m_<mode>_<tracks>_seed<seed>.csv       how much data is left after each step
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from streams import stream_labels  # noqa: E402  (the team's shared stream rule)

EVENTS = ROOT / "data" / "raw" / "yambda" / "flat" / "50m" / "multi_event.parquet"
EMB_IDS = ROOT / "data" / "processed" / "emb50m_item_ids.npy"

REAL_PCT = 50                 # D1
WARMUP_DAYS = 60              # D2
SESSION_GAP_S = 30 * 60       # D3, timestamps are in seconds
MIN_TRACKS = 100              # D5
CAP = 15                      # D4
MAX_BASKETS_PER_USER = 100    # D6
STREAM_NAMES = {1: "organic", 0: "recommended"}

FUNNEL = []


def settings_name(mode, tracks, seed):
    """The settings as they appear in the output file names, e.g. 'discovery_embedded_seed42'."""
    return f"{mode}_{tracks}_seed{seed}"


def baskets_path(mode, tracks, seed):
    return ROOT / "data" / "processed" / f"rq3_baskets_50m_{settings_name(mode, tracks, seed)}.parquet"


def log(step, **numbers):
    """Print one line of the data funnel and remember it for the CSV."""
    FUNNEL.append({"step": step, **numbers})
    print(f"{step:<50}" + "   ".join(f"{k} {v:,}" for k, v in numbers.items()), flush=True)


def load_listens():
    """All listen events. The file is stored sorted by user, then time; we check that, because sessions rely on it."""
    cols = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "event_type"]
    df = pq.read_table(EVENTS, columns=cols).to_pandas()
    df = df[df["event_type"] == "listen"].drop(columns="event_type").reset_index(drop=True)
    uid, ts = df["uid"].to_numpy(), df["timestamp"].to_numpy()
    assert np.all((uid[1:] > uid[:-1]) | ((uid[1:] == uid[:-1]) & (ts[1:] >= ts[:-1]))), "not sorted by user, time"
    return df


def add_stream_and_session(df, mode):
    """Stream label (1 organic, 0 recommended, -1 excluded) and session id of every listen.

    Both are computed on ALL listens, before any filtering, exactly as in the notebooks: the stream rule needs the
    full history of each (user, track), and a session gap is a gap between any two listens.
    """
    uid = df["uid"].to_numpy()
    ts = df["timestamp"].to_numpy().astype(np.int64)
    new_user = np.r_[True, uid[1:] != uid[:-1]]
    uidx = np.cumsum(new_user) - 1                     # compact user index 0..n_users-1, required by stream_labels
    stream = stream_labels(uidx, df["item_id"].to_numpy(), df["is_organic"].to_numpy(),
                           df["played_ratio_pct"].to_numpy(), ts, REAL_PCT, mode, WARMUP_DAYS)
    new_session = new_user | np.r_[True, np.diff(ts) > SESSION_GAP_S]
    return df.assign(stream=stream, session=np.cumsum(new_session) - 1)


def main():
    parser = argparse.ArgumentParser(description="Build the session baskets for RQ3 Method A.")
    parser.add_argument("--mode", default="discovery", choices=["discovery", "listen"], help="stream rule")
    parser.add_argument("--tracks", default="embedded", choices=["embedded", "all"],
                        help="'embedded' keeps only tracks with an audio embedding (D8), 'all' skips that filter")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random choices in D6 and D7")
    args = parser.parse_args()
    if args.tracks == "embedded" and not EMB_IDS.exists():
        sys.exit(f"Missing {EMB_IDS.relative_to(ROOT)} (3 MB list of track ids with an embedding; get it from a "
                 "teammate). Until then run with --tracks all.")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- listens: label, session, filters D1, D2, D8
    df = load_listens()
    log("all listens", listens=len(df), users=df["uid"].nunique())
    df = add_stream_and_session(df, args.mode)
    df = df[df["played_ratio_pct"] >= REAL_PCT]
    log("D1 real listens (>= 50% played)", listens=len(df), users=df["uid"].nunique())
    df = df[df["stream"] >= 0]
    log("D2 stream label known", listens=len(df), users=df["uid"].nunique())
    if args.tracks == "embedded":
        df = df[np.isin(df["item_id"].to_numpy(), np.load(EMB_IDS))]
        log("D8 track has an embedding", listens=len(df), users=df["uid"].nunique())
    else:
        print("D8 skipped (--tracks all): tracks without an embedding are kept")

    # ---- D5: users with at least MIN_TRACKS distinct tracks in both streams
    tracks = df.drop_duplicates(["uid", "stream", "item_id"]).groupby(["uid", "stream"]).size().unstack(fill_value=0)
    eligible = tracks.index[(tracks[0] >= MIN_TRACKS) & (tracks[1] >= MIN_TRACKS)]
    df = df[df["uid"].isin(eligible)]
    log(f"D5 users with >= {MIN_TRACKS} tracks in both streams", listens=len(df), users=len(eligible))

    # ---- D4: baskets = distinct tracks per (session, stream), first CAP in time, at least 2
    df = df.drop_duplicates(["session", "stream", "item_id"])          # keeps the first play of a track
    df = df.assign(position=df.groupby(["session", "stream"]).cumcount())
    df = df[df["position"] < CAP]
    baskets = (df.groupby(["session", "stream"])
                 .agg(uid=("uid", "first"), size=("item_id", "size"))
                 .reset_index())
    baskets = baskets[baskets["size"] >= 2]
    log("D4 baskets with >= 2 tracks", users=baskets["uid"].nunique(),
        organic=int((baskets["stream"] == 1).sum()), recommended=int((baskets["stream"] == 0).sum()))

    # ---- D6: at most MAX_BASKETS_PER_USER random baskets per user and stream
    baskets = baskets.sample(frac=1, random_state=rng)                  # random order
    baskets = baskets[baskets.groupby(["uid", "stream"]).cumcount() < MAX_BASKETS_PER_USER]
    log(f"D6 at most {MAX_BASKETS_PER_USER} baskets per user and stream", users=baskets["uid"].nunique(),
        organic=int((baskets["stream"] == 1).sum()), recommended=int((baskets["stream"] == 0).sum()))

    # ---- D7: the same number of baskets of every size in both streams
    matched = []
    for _, group in baskets.groupby("size"):                         # one group per basket size
        organic, recommended = group[group["stream"] == 1], group[group["stream"] == 0]
        n = min(len(organic), len(recommended))
        matched += [organic.sample(n, random_state=rng), recommended.sample(n, random_state=rng)]
    baskets = pd.concat(matched)
    log("D7 matched basket sizes", users=baskets["uid"].nunique(),
        organic=int((baskets["stream"] == 1).sum()), recommended=int((baskets["stream"] == 0).sum()))

    # ---- one row per track of a chosen basket
    baskets = baskets.sort_values(["stream", "size", "session"]).reset_index(drop=True)
    baskets["basket_id"] = np.arange(len(baskets))
    out = df.merge(baskets[["session", "stream", "basket_id", "size"]], on=["session", "stream"])
    out["stream"] = out["stream"].map(STREAM_NAMES)
    out = out.sort_values(["basket_id", "position"])[["basket_id", "uid", "stream", "size", "position", "item_id"]]

    # ---- checks
    assert not out.duplicated(["basket_id", "item_id"]).any(), "a track appears twice in one basket"
    assert (out.groupby("basket_id").size() == out.groupby("basket_id")["size"].first()).all(), "wrong basket size"
    per_size = out.drop_duplicates("basket_id").groupby(["size", "stream"]).size().unstack(fill_value=0)
    assert (per_size["organic"] == per_size["recommended"]).all(), "basket sizes differ between the streams"

    path = baskets_path(args.mode, args.tracks, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    funnel_path = ROOT / "outputs" / f"rq3_baskets_funnel_50m_{settings_name(args.mode, args.tracks, args.seed)}.csv"
    funnel_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(FUNNEL).to_csv(funnel_path, index=False)

    n_baskets = out["basket_id"].nunique()
    print(f"\n{n_baskets // 2:,} baskets per stream, mean size {len(out) / n_baskets:.2f} tracks, "
          f"{out['item_id'].nunique():,} distinct tracks, {out['uid'].nunique():,} users  ({time.time() - t0:.0f} s)")
    print("basket sizes (same in both streams):", per_size["organic"].to_dict())
    print(f"saved {path.relative_to(ROOT)} and {funnel_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
