"""Frequent pairs of items: naive counting, A-Priori and PCY.

Follows the week 5 lecture (Computational Tools for Data Science) and MMDS chapter 6.
Only the Python standard library is used.

Conventions
- A basket is a collection of distinct items. Items are integers (lecture slide 27: use integers, not names),
  for us the Yambda track ids.
- A pair is always stored as (a, b) with a < b, so it is counted in exactly one place.
- min_count is the support threshold as a number of baskets: an itemset is frequent if it occurs in at least
  min_count baskets. Use min_count_for() to turn a relative support s (lecture slide 7) into a count.
- Pair counts are stored with the triples method, i.e. a dict {(a, b): count} (lecture slide 30). Our baskets are
  very sparse (far fewer than 1/3 of all possible pairs ever occur), which is the case where triples win over the
  triangular matrix. PCY needs triples anyway (slide 45).

Every algorithm returns (frequent_pairs, info): frequent_pairs is {(a, b): count} for all pairs with
count >= min_count, and info is a small dict, e.g. how many candidate pairs had to be counted.
"""

import math
from collections import Counter
from itertools import combinations

NUM_BUCKETS = 2 ** 20  # PCY: number of buckets in the hash table of pass 1


def min_count_for(support, n_baskets):
    """Smallest number of baskets that reaches the relative support threshold s (support >= s)."""
    return math.ceil(round(support * n_baskets, 9))  # round() avoids float noise such as 0.1 * 30 = 3.0000000000000004


def count_items(baskets):
    """Number of baskets that contain each item (the counts of all singleton itemsets)."""
    counts = Counter()
    for basket in baskets:
        counts.update(set(basket))
    return counts


def naive_pairs(baskets, min_count):
    """Naive algorithm (lecture slide 31): one pass, count every pair of every basket.

    Needs memory for every pair that occurs at least once; we use it only as the reference to check the others.
    """
    pair_counts = Counter()
    for basket in baskets:
        pair_counts.update(combinations(sorted(set(basket)), 2))
    frequent = {pair: c for pair, c in pair_counts.items() if c >= min_count}
    return frequent, {"candidates": len(pair_counts)}


def apriori_pairs(baskets, min_count):
    """A-Priori for pairs (lecture slides 34-36), two passes over the baskets.

    Pass 1: count every single item.
    Between the passes: the frequent-items table. (The lecture renumbers the frequent items 1..m so that a triangular
             matrix fits; with the triples method a set of the frequent items is all we need.)
    Pass 2: count only pairs whose two items are both frequent. By monotonicity (slide 33) a pair with an infrequent
             item cannot be frequent, so no frequent pair is missed.
    """
    item_counts = count_items(baskets)                                        # pass 1
    frequent_items = {i for i, c in item_counts.items() if c >= min_count}    # between the passes

    pair_counts = Counter()                                                   # pass 2
    for basket in baskets:
        items = sorted(i for i in set(basket) if i in frequent_items)
        pair_counts.update(combinations(items, 2))

    frequent = {pair: c for pair, c in pair_counts.items() if c >= min_count}
    return frequent, {"candidates": len(pair_counts), "frequent_items": len(frequent_items)}


def bucket_of(a, b, num_buckets):
    """PCY hash function: the bucket of the pair (a, b)."""
    return (a * 1_000_003 + b) % num_buckets


def pcy_pairs(baskets, min_count, num_buckets=NUM_BUCKETS):
    """PCY algorithm for pairs (lecture slides 40-45), two passes over the baskets.

    Pass 1: as A-Priori, count every single item. In addition, generate every pair of the basket, hash it to a
             bucket and add 1 to that bucket's count. The pair itself is not stored.
    Between the passes: frequent items, and a bitmap of the frequent buckets (count >= min_count). A bucket that
             is not frequent cannot contain a frequent pair, since its count is the sum of the counts of its pairs.
    Pass 2: count a pair only if (a) both items are frequent and (b) the pair hashes to a frequent bucket.

    The result is exactly the same as A-Priori; only fewer candidate pairs are counted.
    """
    item_counts = Counter()                                                   # pass 1
    bucket_counts = [0] * num_buckets
    for basket in baskets:
        items = sorted(set(basket))
        item_counts.update(items)
        for a, b in combinations(items, 2):
            bucket_counts[bucket_of(a, b, num_buckets)] += 1

    frequent_items = {i for i, c in item_counts.items() if c >= min_count}    # between the passes
    bitmap = bytearray(c >= min_count for c in bucket_counts)                 # 1 byte per bucket (a real bitmap uses 1 bit)
    del bucket_counts

    pair_counts = Counter()                                                   # pass 2
    for basket in baskets:
        items = sorted(i for i in set(basket) if i in frequent_items)
        for a, b in combinations(items, 2):
            if bitmap[bucket_of(a, b, num_buckets)]:
                pair_counts[(a, b)] += 1

    frequent = {pair: c for pair, c in pair_counts.items() if c >= min_count}
    return frequent, {"candidates": len(pair_counts), "frequent_items": len(frequent_items),
                      "frequent_bucket_share": sum(bitmap) / num_buckets}


def pair_measures(frequent_pairs, item_counts, n_baskets):
    """Support, confidence (both directions) and lift of every frequent pair (lecture slides 7, 8, 21).

    support(a, b)   = count(a, b) / N
    confidence(a->b) = count(a, b) / count(a)
    lift(a->b)      = confidence(a->b) / support(b) = count(a, b) * N / (count(a) * count(b))
    For a pair the lift is the same in both directions. Lift close to 1: the two items occur together as often as
    expected by chance; lift >> 1: they occur together much more often (slide 22).
    """
    rows = []
    for (a, b), c in frequent_pairs.items():
        rows.append({
            "a": a,
            "b": b,
            "count": c,
            "support": c / n_baskets,
            "conf_ab": c / item_counts[a],
            "conf_ba": c / item_counts[b],
            "lift": c * n_baskets / (item_counts[a] * item_counts[b]),
        })
    return rows
