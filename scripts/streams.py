"""Which stream (organic or recommended) a listen belongs to. Shared by all notebooks and scripts.

Default rule ("discovery"): every (user, track) pair belongs to exactly one stream, the stream of the
user's first real listen of that track. A track the recommender introduced stays "recommended" even when
the user replays it later by choice. Tracks the user already played in their first WARMUP days of
activity are excluded (-1), because the log starts mid-history and those tracks may be old favourites.

Usage (arrays have one value per listen, sorted by user, then time):
    from streams import stream_labels
    lab = stream_labels(uidx, item, is_organic, played_pct, ts, real_pct=50)
    # lab: 1 = organic, 0 = recommended, -1 = excluded
"""

import numpy as np

MODES = ("discovery", "discovery_any", "listen")
DAY_S = 86400


def stream_labels(uidx, item, org, ratio, ts, real_pct, mode="discovery", warmup_days=60):
    """Stream label per listen: 1 = organic, 0 = recommended, -1 = excluded.

    mode="discovery":     is_organic of the first real listen (>= real_pct played) of each (user, track);
                          pairs without a real listen get -1.
    mode="discovery_any": is_organic of the first listen of any length of each (user, track).
    mode="listen":        is_organic of each listen itself (the old rule; no warm-up).
    warmup_days:          (user, track) pairs with any listen in the user's first warmup_days days get -1
                          (discovery modes only; 0 switches it off).
    uidx must be a compact user index (0..n_users-1) and the rows sorted by user, then time.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    org = np.asarray(org).astype(np.int8)
    if mode == "listen":
        return org.copy()

    key = (np.asarray(uidx).astype(np.int64) << 32) | np.asarray(item).astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    n_pairs = int(inv.max()) + 1
    del key

    # first qualifying listen of every pair (rows are in time order within a user)
    q = np.flatnonzero(np.asarray(ratio) >= real_pct) if mode == "discovery" else np.arange(len(inv))
    pairs, first = np.unique(inv[q], return_index=True)
    label = np.full(n_pairs, -1, dtype=np.int8)
    label[pairs] = org[q[first]]

    if warmup_days > 0:
        ts = np.asarray(ts).astype(np.int64)
        uidx = np.asarray(uidx)
        starts = np.r_[0, np.flatnonzero(uidx[1:] != uidx[:-1]) + 1]
        user_start = np.empty(int(uidx.max()) + 1, dtype=np.int64)
        user_start[uidx[starts]] = np.minimum.reduceat(ts, starts)
        in_warmup = ts < user_start[uidx] + warmup_days * DAY_S
        label[np.unique(inv[in_warmup])] = -1

    return label[inv]
