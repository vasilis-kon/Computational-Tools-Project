"""Day-one profile of Yambda-50M: what can and cannot we do with this data?

Reads data/raw/yambda/ and writes a text report to outputs/day1_profile.txt.
Run (works from any folder):
    python scripts/profile_yambda.py

Memory: about 3-4 GB at the peak. Runtime: a few minutes.

Definitions used here (change them at the top if the team decides otherwise):
    real listen = played_ratio_pct >= REAL_LISTEN_PCT
    session     = listens of one user with gaps of at most SESSION_GAP_S
    stream      = organic (is_organic == 1) or recommended (is_organic == 0)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REAL_LISTEN_PCT = 50
SESSION_GAP_S = 30 * 60      # timestamps are in seconds (binned to 5 s)
MIN_REAL = 50                # "enough listens" threshold for per-user comparisons

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "yambda"
REPORT = ROOT / "outputs" / "day1_profile.txt"

lines = []


def out(text=""):
    print(text, flush=True)
    lines.append(str(text))


def head(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


def pct(x, total):
    return f"{100 * x / total:.2f}%"


def quantiles(arr, name):
    qs = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    vals = np.quantile(arr, qs)
    out(f"{name}: " + "  ".join(f"p{int(q * 100)}={v:,.1f}" for q, v in zip(qs, vals)))


def per_user_count(user_idx, mask, n_users):
    """Number of rows per user where mask is True."""
    return np.bincount(user_idx[mask], minlength=n_users)


# ---------------------------------------------------------------- load
t0 = time.time()
out(f"Yambda-50M profile  (real listen >= {REAL_LISTEN_PCT}% played, session gap {SESSION_GAP_S // 60} min)")
df = pq.read_table(DATA / "flat/50m/multi_event.parquet").to_pandas()
uid_all = df["uid"].to_numpy()
is_org_all = df["is_organic"].to_numpy()
etype = df["event_type"]
categories = list(etype.cat.categories)

# ---------------------------------------------------------------- A. overview
head("A. Overview of multi_event")
n_rows = len(df)
out(f"rows: {n_rows:,}")
out(f"users: {len(np.unique(uid_all)):,}   distinct items: {df['item_id'].nunique():,}")
out(f"timestamp range: {df['timestamp'].min():,} .. {df['timestamp'].max():,} s "
    f"= {df['timestamp'].max() / 86400:.0f} days")
out("rows sorted by (uid, timestamp)? " + str(
    bool(np.all((uid_all[1:] > uid_all[:-1]) |
                ((uid_all[1:] == uid_all[:-1]) & (df['timestamp'].to_numpy()[1:] >= df['timestamp'].to_numpy()[:-1]))))))
out("\nEvent type x is_organic (rows):")
ct = pd.crosstab(etype, is_org_all)
ct.columns = ["is_organic=0", "is_organic=1"]
ct["share is_organic=1"] = (ct["is_organic=1"] / ct.sum(axis=1)).map("{:.1%}".format)
out(ct.to_string())

# From here on: listens only, as plain numpy arrays.
lm = (etype == "listen").to_numpy()
likes_by_org = np.bincount(is_org_all[(etype == "like").to_numpy()], minlength=2)
dislikes_by_org = np.bincount(is_org_all[(etype == "dislike").to_numpy()], minlength=2)
uid = uid_all[lm]
item = df["item_id"].to_numpy()[lm]
ts = df["timestamp"].to_numpy()[lm].astype(np.int64)
org = is_org_all[lm]
ratio = df["played_ratio_pct"].to_numpy()[lm]
length = df["track_length_seconds"].to_numpy()[lm]
del df, uid_all, is_org_all, etype
n = len(uid)
real = ratio >= REAL_LISTEN_PCT
n_ratio_null = int(np.isnan(ratio).sum()) if ratio.dtype.kind == "f" else 0

# compact user index 0..n_users-1 (data is sorted by uid, so use change points)
new_user = np.empty(n, dtype=bool)
new_user[0] = True
new_user[1:] = uid[1:] != uid[:-1]
uidx = np.cumsum(new_user) - 1
n_users = int(uidx[-1]) + 1

# ---------------------------------------------------------------- B. is_organic
head("B. Is the is_organic coding what we think? (compare behaviour of the two values)")
out("The dataset card says is_organic=1 means organic, 0 means recommendation-driven.")
out("If that is right, recommended listens should be skipped more and liked less.\n")
rows = []
for value, name in [(1, "organic (1)"), (0, "recommended (0)")]:
    m = org == value
    r = ratio[m]
    rows.append({
        "stream": name,
        "listens": f"{m.sum():,}",
        "share": pct(m.sum(), n),
        "mean played %": f"{np.nanmean(r):.1f}",
        "played <30%": pct(np.sum(r < 30), m.sum()),
        f"played >={REAL_LISTEN_PCT}%": pct(np.sum(r >= REAL_LISTEN_PCT), m.sum()),
        "played >=80%": pct(np.sum(r >= 80), m.sum()),
        "played >100%": pct(np.sum(r > 100), m.sum()),
        "likes / 1000 listens": f"{1000 * likes_by_org[value] / m.sum():.2f}",
        "dislikes / 1000 listens": f"{1000 * dislikes_by_org[value] / m.sum():.2f}",
        "mean length s": f"{length[m].mean():.0f}",
    })
out(pd.DataFrame(rows).set_index("stream").T.to_string())
out(f"\nlistens with missing played_ratio_pct: {n_ratio_null:,}")

# ---------------------------------------------------------------- C. users
head("C. Users: how much data does each user have in each stream?")
listens_pu = np.bincount(uidx, minlength=n_users)
org_pu = per_user_count(uidx, org == 1, n_users)
rec_pu = listens_pu - org_pu
real_org_pu = per_user_count(uidx, real & (org == 1), n_users)
real_rec_pu = per_user_count(uidx, real & (org == 0), n_users)
quantiles(listens_pu, "listens per user       ")
quantiles(org_pu / listens_pu, "organic share per user  ")
quantiles(real_org_pu, f"real organic listens/user")
quantiles(real_rec_pu, f"real recommended listens/user")
both = (real_org_pu >= MIN_REAL) & (real_rec_pu >= MIN_REAL)
out(f"\nusers with >= {MIN_REAL} real listens in BOTH streams: {both.sum():,} of {n_users:,} ({pct(both.sum(), n_users)})")
for thr in (0.05, 0.10):
    low = (org_pu / listens_pu) < thr
    high = (org_pu / listens_pu) > 1 - thr
    out(f"users with <{thr:.0%} organic: {low.sum():,}   users with >{1 - thr:.0%} organic: {high.sum():,}")
share = org_pu / listens_pu
band_edges = [0, 0.25, 0.5, 0.75, 1.0001]
band = np.digitize(share, band_edges[1:-1])
out("\nUsers by organic share band (for 'compare within the same share band'):")
for b in range(4):
    out(f"  organic share {band_edges[b]:.0%}-{min(band_edges[b + 1], 1):.0%}: {(band == b).sum():,} users")

# ---------------------------------------------------------------- D. items / popularity / discovery
head("D. Items: popularity, overlap between streams, and discovery")
max_item = int(item.max()) + 1
pop = np.bincount(item, minlength=max_item)
pop_org = np.bincount(item[org == 1], minlength=max_item)
pop_rec = np.bincount(item[org == 0], minlength=max_item)
n_items = int((pop > 0).sum())
out(f"distinct items listened: {n_items:,}   (organic: {(pop_org > 0).sum():,}, recommended: {(pop_rec > 0).sum():,})")
only_org = ((pop_org > 0) & (pop_rec == 0)).sum()
only_rec = ((pop_rec > 0) & (pop_org == 0)).sum()
out(f"items only ever played organically: {only_org:,}   only ever via recommendation: {only_rec:,}   both: {n_items - only_org - only_rec:,}")
top = np.argsort(pop)[::-1][: max(1, n_items // 100)]
out(f"\nPopularity concentration: share of each stream's listens that goes to the top 1% most popular items ({len(top):,} items):")
out(f"  organic: {pct(pop_org[top].sum(), pop_org.sum())}   recommended: {pct(pop_rec[top].sum(), pop_rec.sum())}")

key = uidx.astype(np.int64) * (1 << 32) + item
uniq, first_idx, inverse = np.unique(key, return_index=True, return_inverse=True)
n_pairs = len(uniq)
out(f"\nDistinct (user, item) pairs: {n_pairs:,}   -> {n / n_pairs:.2f} listens per pair (repeat listening)")
first_org = org[first_idx]
out(f"Pairs whose FIRST listen was organic: {pct(first_org.sum(), n_pairs)}   recommended: {pct((first_org == 0).sum(), n_pairs)}")
has_org = np.bincount(inverse, weights=(org == 1), minlength=n_pairs) > 0
has_rec = np.bincount(inverse, weights=(org == 0), minlength=n_pairs) > 0
out(f"Pairs heard in both streams: {pct((has_org & has_rec).sum(), n_pairs)}  "
    f"(organic only: {pct((has_org & ~has_rec).sum(), n_pairs)}, recommended only: {pct((~has_org & has_rec).sum(), n_pairs)})")
out("-> Design decision: a track found via a recommendation and replayed later counts as 'organic'")
out("   in the second case. Decide whether the first discovery or each play counts.")
repeat_org = 1 - len(np.unique(key[org == 1])) / (org == 1).sum()
repeat_rec = 1 - len(np.unique(key[org == 0])) / (org == 0).sum()
out(f"Repeat share within a stream (listens that are not the first of that user-item pair in that stream): "
    f"organic {repeat_org:.1%}, recommended {repeat_rec:.1%}")
del uniq, first_idx, inverse, key, has_org, has_rec

# ---------------------------------------------------------------- E. sessions
head(f"E. Sessions (gap > {SESSION_GAP_S // 60} min starts a new session) -> basket sizes for frequent itemsets")
new_sess = np.empty(n, dtype=bool)
new_sess[0] = True
new_sess[1:] = new_user[1:] | ((ts[1:] - ts[:-1]) > SESSION_GAP_S)
sid = np.cumsum(new_sess) - 1
n_sess = int(sid[-1]) + 1
size = np.bincount(sid, minlength=n_sess)
real_org_s = np.bincount(sid[real & (org == 1)], minlength=n_sess)
real_rec_s = np.bincount(sid[real & (org == 0)], minlength=n_sess)
out(f"sessions: {n_sess:,}   per user: median {np.median(np.bincount(uidx[new_sess], minlength=n_users)):.0f}")
quantiles(size, "listens per session            ")
quantiles(real_org_s + real_rec_s, "real listens per session       ")
mixed = (real_org_s > 0) & (real_rec_s > 0)
out(f"\nsessions with real listens from both streams: {pct(mixed.sum(), n_sess)}")
out(f"sessions with only organic real listens: {pct(((real_org_s > 0) & (real_rec_s == 0)).sum(), n_sess)}   "
    f"only recommended: {pct(((real_rec_s > 0) & (real_org_s == 0)).sum(), n_sess)}")
out(f"sessions with >= 2 real organic listens: {(real_org_s >= 2).sum():,}   >= 2 real recommended: {(real_rec_s >= 2).sum():,}")
big = size > 100
out(f"sessions with > 100 listens: {big.sum():,} ({pct(big.sum(), n_sess)})  -> pair counting needs a cap or thresholds")
quantiles(real_org_s[real_org_s > 0], "real organic per session (>0)")
quantiles(real_rec_s[real_rec_s > 0], "real recommended per session (>0)")
out("Candidate pairs per session ~ k*(k-1)/2: sum over sessions of the real listens of each stream:")
out(f"  organic: {int((real_org_s.astype(np.int64) * (real_org_s - 1) // 2).sum()):,}   "
    f"recommended: {int((real_rec_s.astype(np.int64) * (real_rec_s - 1) // 2).sum()):,}")

# ---------------------------------------------------------------- F. artists / albums / embeddings
head("F. Artist, album and embedding coverage (the only 'content' information we have)")
art = pd.read_parquet(DATA / "artist_item_mapping.parquet")
alb = pd.read_parquet(DATA / "album_item_mapping.parquet")
for name, mp, col in [("artist", art, "artist_id"), ("album", alb, "album_id")]:
    per_item = mp.groupby("item_id")[col].nunique()
    out(f"{name} mapping: {len(mp):,} rows, {mp['item_id'].nunique():,} items, {mp[col].nunique():,} {name}s, "
        f"items with >1 {name}: {(per_item > 1).sum():,}")
    covered = np.zeros(max_item, dtype=bool)
    ids = mp["item_id"].to_numpy()
    covered[ids[ids < max_item]] = True
    out(f"   covers {pct(covered[pop > 0].sum(), n_items)} of listened items and "
        f"{pct(pop[covered].sum(), n)} of listens")
# first artist per item (an item with several artists is assigned to one of them)
art1 = art.drop_duplicates("item_id")
item_art = np.full(max_item, -1, dtype=np.int64)
a_items = art1["item_id"].to_numpy()
keep = a_items < max_item
item_art[a_items[keep]] = art1["artist_id"].to_numpy()[keep]
a = item_art[item]
has_a = a >= 0
out("\nArtists per stream (real listens with an artist):")
for value, name in [(1, "organic"), (0, "recommended")]:
    m = has_a & real & (org == value)
    arts = a[m]
    cnt = np.unique(arts, return_counts=True)[1]
    top1 = np.sort(cnt)[::-1][: max(1, len(cnt) // 100)].sum() / cnt.sum()
    # distinct artists per user (users with at least MIN_REAL real listens in this stream)
    k = uidx[m].astype(np.int64) * (1 << 32) + arts
    pu = np.bincount((np.unique(k) >> 32).astype(np.int64), minlength=n_users)
    n_real_stream = np.bincount(uidx[m], minlength=n_users)
    ok = n_real_stream >= MIN_REAL
    out(f"  {name}: {len(cnt):,} distinct artists, top 1% artists get {top1:.1%} of listens; "
        f"distinct artists per 100 listens (median over {ok.sum():,} users): "
        f"{np.median(100 * pu[ok] / n_real_stream[ok]):.1f}")

emb_path = DATA / "embeddings.parquet"
EMB_SIZE = 13_814_230_943   # full size, see download_yambda.py
if emb_path.exists() and emb_path.stat().st_size == EMB_SIZE:
    pf = pq.ParquetFile(emb_path)
    emb_ids = pf.read(columns=["item_id"]).column("item_id").to_numpy()
    covered = np.zeros(max_item, dtype=bool)
    covered[emb_ids[emb_ids < max_item]] = True
    out(f"\nembeddings: {pf.metadata.num_rows:,} items; cover {pct(covered[pop > 0].sum(), n_items)} of listened items "
        f"and {pct(pop[covered].sum(), n)} of listens (organic: {pct(pop_org[covered].sum(), pop_org.sum())}, "
        f"recommended: {pct(pop_rec[covered].sum(), pop_rec.sum())})")
    first = next(pf.iter_batches(batch_size=1, columns=["embed"])).column("embed")[0].as_py()
    out(f"embedding dimension: {len(first)}")
else:
    out("\nembeddings.parquet missing or not fully downloaded yet "
        "(run download_yambda.py --embeddings); skipped.")

# ---------------------------------------------------------------- G. time
head("G. Time: is there enough history per user to compare early and late (RQ6)?")
span_days = ts.max() / 86400
third = ts.max() / 3
for lo_name, lo_mask in [("first third", ts < third), ("last third", ts >= 2 * third)]:
    c_org = per_user_count(uidx, lo_mask & real & (org == 1), n_users)
    c_rec = per_user_count(uidx, lo_mask & real & (org == 0), n_users)
    out(f"{lo_name} of the {span_days:.0f} days: users with >= {MIN_REAL} real organic listens: "
        f"{(c_org >= MIN_REAL).sum():,}   with >= {MIN_REAL} real recommended: {(c_rec >= MIN_REAL).sum():,}")
f_org = per_user_count(uidx, (ts < third) & real & (org == 1), n_users)
l_org = per_user_count(uidx, (ts >= 2 * third) & real & (org == 1), n_users)
out(f"users with >= {MIN_REAL} real organic listens in BOTH first and last third: {((f_org >= MIN_REAL) & (l_org >= MIN_REAL)).sum():,}")
first_ts = np.minimum.reduceat(ts, np.flatnonzero(new_user))
last_ts = np.maximum.reduceat(ts, np.flatnonzero(new_user))
quantiles((last_ts - first_ts) / 86400, "days between a user's first and last listen")
out("Note: timestamps are seconds since the dataset start (5 s bins), not calendar time.")

out(f"\nDone in {time.time() - t0:.0f} s")
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"Report saved to {REPORT.relative_to(ROOT)}")
