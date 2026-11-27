"""Steps 3 and 4 of RQ3 Method A: frequent track pairs per stream, over a range of support thresholds (the "sweep"),
and the hand-off file for Method B at the main threshold MAIN_SUPPORT.

For every stream (organic, recommended) and every support threshold s in SUPPORTS:
  - run A-Priori and PCY and check that they find exactly the frequent pairs of the naive count
    (correctness on the real data, not only on the test data of check_itemsets.py)
  - record how many candidate pairs each algorithm had to count and how long it took (cost, useful for RQ6)
  - describe the result: frequent tracks, frequent pairs, pairs played by at least MIN_USERS different users (D13),
    their median lift, the share of baskets that contain such a pair, and how many pairs both streams share

The sweep is how we chose the main threshold s = 0.0001 (at least 26 of 258,897 baskets): high enough that a pair is a
real pattern, low enough that both streams still have pairs to compare and to hand to Method B.

Run from the project root, after baskets.py (about half a minute):
    python rq3_method_a/run.py --tracks all
Output:
    outputs/rq3_method_a_sweep_50m_<settings>.csv              the sweep table
    data/processed/rq3_method_a_pairs_50m_<settings>.parquet   the frequent pairs at MAIN_SUPPORT with >= MIN_USERS users
    data/processed/rq3_method_a_pairs_50m.parquet              the same file under the name Method B reads; written only
                                                               for the main configuration (mode discovery, seed 42)
Columns of the pairs file: stream, a, b, lift (the four Method B needs), count, support, conf_ab, conf_ba, n_users,
and the settings (level, support_threshold, min_count, min_users, stream_mode, tracks, seed, run_date).
"""

import argparse
import sys
import time
from datetime import date
from itertools import combinations
from statistics import median

import pandas as pd

from baskets import ROOT, baskets_path, settings_name
from itemsets import apriori_pairs, count_items, min_count_for, naive_pairs, pair_measures, pcy_pairs

SUPPORTS = [0.00002, 0.00005, 0.0001, 0.0002, 0.0005]   # relative support thresholds s (lecture slide 7)
MAIN_SUPPORT = 0.0001                                    # D11: chosen from the sweep
MIN_USERS = 5                                            # D13: a pair must be played by at least 5 different users
PAIR_COLUMNS = ["stream", "a", "b", "lift", "count", "support", "conf_ab", "conf_ba", "n_users"]
assert MAIN_SUPPORT in SUPPORTS


def load_baskets(path):
    """{stream: (baskets, uids)}: baskets is a list of lists of track ids, uids the user of each basket."""
    df = pd.read_parquet(path).sort_values(["basket_id", "position"])
    data = {}
    for stream, g in df.groupby("stream"):
        items = g["item_id"].tolist()                      # plain Python ints
        sizes = g.groupby("basket_id").size().tolist()
        baskets, start = [], 0
        for n in sizes:
            baskets.append(items[start:start + n])
            start += n
        data[stream] = (baskets, g.drop_duplicates("basket_id")["uid"].tolist())
    return data


def users_per_pair(baskets, uids, pairs):
    """Number of different users whose baskets contain each pair (only for the given pairs)."""
    users = {pair: set() for pair in pairs}
    for basket, uid in zip(baskets, uids):
        for pair in combinations(sorted(basket), 2):
            if pair in users:
                users[pair].add(uid)
    return {pair: len(u) for pair, u in users.items()}


def best_pair_count(baskets, pair_counts):
    """For every basket, the highest count among the pairs it contains (0 if it contains none of them).

    A basket contains a frequent pair at threshold min_count exactly when this number is >= min_count, so one pass
    gives the "share of baskets with a frequent pair" for every threshold.
    """
    best = []
    for basket in baskets:
        counts = [pair_counts[p] for p in combinations(sorted(basket), 2) if p in pair_counts]
        best.append(max(counts, default=0))
    return best


def analyse_stream(stream, baskets, uids):
    n = len(baskets)
    item_counts = count_items(baskets)

    t = time.time()
    all_pairs, naive_info = naive_pairs(baskets, 1)        # min_count 1 = the count of every pair that occurs
    sec_naive = time.time() - t
    lowest = min_count_for(min(SUPPORTS), n)
    kept = {p: c for p, c in all_pairs.items() if c >= lowest}   # every pair that is frequent at some threshold
    del all_pairs

    n_users = users_per_pair(baskets, uids, kept)
    strong = {p: c for p, c in kept.items() if n_users[p] >= MIN_USERS}
    best = best_pair_count(baskets, strong)

    rows, pair_sets = [], {}
    for s in SUPPORTS:
        mc = min_count_for(s, n)
        t = time.time()
        ap, ap_info = apriori_pairs(baskets, mc)
        sec_ap = time.time() - t
        t = time.time()
        pcy, pcy_info = pcy_pairs(baskets, mc)
        sec_pcy = time.time() - t
        reference = {p: c for p, c in kept.items() if c >= mc}
        assert ap == reference, f"{stream}, s={s}: A-Priori differs from the naive count"
        assert pcy == reference, f"{stream}, s={s}: PCY differs from the naive count"

        frequent_strong = {p: c for p, c in strong.items() if c >= mc}
        measures = pair_measures(frequent_strong, item_counts, n)
        lifts = [r["lift"] for r in measures]
        pair_sets[s] = set(frequent_strong)
        if s == MAIN_SUPPORT:                               # the pairs for the hand-off file
            main_pairs = [{**r, "stream": stream, "n_users": n_users[(r["a"], r["b"])], "min_count": mc}
                          for r in measures]
        rows.append({
            "stream": stream, "support": s, "min_count": mc, "baskets": n,
            "frequent_tracks": ap_info["frequent_items"],
            "frequent_pairs": len(ap),
            f"pairs_{MIN_USERS}plus_users": len(frequent_strong),
            "median_lift": median(lifts) if lifts else float("nan"),
            "baskets_with_pair": sum(b >= mc for b in best) / n,
            "cand_naive": naive_info["candidates"], "cand_apriori": ap_info["candidates"],
            "cand_pcy": pcy_info["candidates"], "pcy_frequent_buckets": pcy_info["frequent_bucket_share"],
            "sec_naive": sec_naive, "sec_apriori": sec_ap, "sec_pcy": sec_pcy,
        })
        print(f"  {stream}, s={s}: done (A-Priori {sec_ap:.1f} s, PCY {sec_pcy:.1f} s)", flush=True)
    return rows, pair_sets, main_pairs


def recount(pairs, data, per_stream=50, seed=0):
    """Independent check of the pairs file: for a random sample of pairs, count directly in the baskets (with plain
    set membership, none of the itemset code) the baskets with a, with b, with both, and the users with both."""
    for stream, (baskets, uids) in data.items():
        sets = [set(b) for b in baskets]
        n = len(sets)
        sample = pairs[pairs["stream"] == stream]
        sample = sample.sample(min(per_stream, len(sample)), random_state=seed)
        for r in sample.to_dict("records"):
            a, b = int(r["a"]), int(r["b"])
            has_a = [a in s for s in sets]
            has_b = [b in s for s in sets]
            both = [i for i in range(n) if has_a[i] and has_b[i]]
            assert len(both) == r["count"], f"{stream} pair ({a}, {b}): count differs"
            assert len({uids[i] for i in both}) == r["n_users"], f"{stream} pair ({a}, {b}): n_users differs"
            assert abs(len(both) * n / (sum(has_a) * sum(has_b)) - r["lift"]) < 1e-9 * r["lift"], "lift differs"


def main():
    parser = argparse.ArgumentParser(description="Frequent track pairs per stream: support sweep.")
    parser.add_argument("--mode", default="discovery", choices=["discovery", "listen"])
    parser.add_argument("--tracks", default="embedded", choices=["embedded", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = baskets_path(args.mode, args.tracks, args.seed)
    if not path.exists():
        sys.exit(f"Missing {path.relative_to(ROOT)}: run rq3_method_a/baskets.py with the same options first.")

    t0 = time.time()
    data = load_baskets(path)
    rows, pair_sets, main_pairs = [], {}, []
    for stream, (baskets, uids) in data.items():
        r, pair_sets[stream], m = analyse_stream(stream, baskets, uids)
        rows += r
        main_pairs += m

    # overlap: share of the frequent pairs (with enough users) that both streams have (Jaccard of the two sets)
    for row in rows:
        o, r = pair_sets["organic"][row["support"]], pair_sets["recommended"][row["support"]]
        row["overlap_jaccard"] = len(o & r) / len(o | r) if o | r else float("nan")

    sweep = pd.DataFrame(rows)
    out = ROOT / "outputs" / f"rq3_method_a_sweep_50m_{settings_name(args.mode, args.tracks, args.seed)}.csv"
    sweep.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print("\nA-Priori and PCY found exactly the frequent pairs of the naive count at every threshold.\n")
    print("Result per stream and support threshold s:")
    print(sweep[["stream", "support", "min_count", "frequent_tracks", "frequent_pairs", f"pairs_{MIN_USERS}plus_users",
                 "median_lift", "baskets_with_pair", "overlap_jaccard"]].to_string(index=False, float_format="{:.4g}".format))
    print("\nCost: candidate pairs counted, share of frequent PCY buckets, seconds:")
    print(sweep[["stream", "support", "cand_naive", "cand_apriori", "cand_pcy", "pcy_frequent_buckets",
                 "sec_naive", "sec_apriori", "sec_pcy"]].to_string(index=False, float_format="{:.3g}".format))
    print(f"\nsaved {out.relative_to(ROOT)}")

    # ---- step 4: the hand-off file for Method B (pairs at MAIN_SUPPORT played by >= MIN_USERS users)
    pairs = pd.DataFrame(main_pairs)
    settings = {"level": "track", "support_threshold": MAIN_SUPPORT, "min_count": pairs["min_count"],
                "min_users": MIN_USERS, "stream_mode": args.mode, "tracks": args.tracks, "seed": args.seed,
                "run_date": date.today().isoformat()}
    pairs = pairs[PAIR_COLUMNS].assign(**settings).sort_values(["stream", "lift"], ascending=[True, False])
    assert (pairs["a"] < pairs["b"]).all() and (pairs["n_users"] >= MIN_USERS).all()
    assert (pairs["count"] >= pairs["min_count"]).all()
    recount(pairs, data)
    print("\nRecount of 50 random pairs per stream directly in the baskets: count, users and lift are correct.")

    files = [ROOT / "data" / "processed" / f"rq3_method_a_pairs_50m_{settings_name(args.mode, args.tracks, args.seed)}.parquet"]
    if args.mode == "discovery" and args.seed == 42:
        files.append(ROOT / "data" / "processed" / "rq3_method_a_pairs_50m.parquet")   # the name Method B reads
    for f in files:
        pairs.to_parquet(f, index=False)
    summary = pairs.groupby("stream").agg(pairs=("lift", "size"), median_lift=("lift", "median"),
                                          median_count=("count", "median"), median_users=("n_users", "median"))
    print(f"\nHand-off at s = {MAIN_SUPPORT} (>= {pairs['min_count'].iloc[0]} baskets, >= {MIN_USERS} users):")
    print(summary.to_string(float_format="{:.1f}".format))
    for f in files:
        print(f"saved {f.relative_to(ROOT)}")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
