# Project notes: do recommendations broaden or narrow listening? (Yambda)

Last updated: 2026-11-08. Hand-in: 4 December 2026.
**To continue in a new session: give Claude this file and say "continue from PROJECT_NOTES.md".**

## 1. Question and data
- Question: for the same users, is what the recommender plays (`is_organic = 0`) broader or narrower than what they choose themselves (`is_organic = 1`)?
- Data: Yambda (Yandex Music), Hugging Face `yandex/yambda`, fixed revision `dd6f3a19eef5866e346c3270e098baa641a44948`. All IDs are anonymous: no names, no genres.
  Content information only through artist/album ID mappings and 128-dim audio embeddings.
- Requirement from the course (as understood): use 2-3 course methods. Confirm the formal brief (report length, whether a distributed framework is required, which outside method).

## 2. Files in this project
| Path | What |
|---|---|
| `scripts/download_yambda.py` | downloads 50M events + artist/album mappings; `--embeddings` adds the 13.8 GB embeddings. Resumable, 30 s timeout, retries, progress. |
| `scripts/download_yambda_500m.py` | NEW: downloads `flat/500m/multi_event.parquet` (3.84 GB); `--check` only compares the size on the server. |
| `scripts/extract_embeddings_50m.py` | keeps the 848k tracks of Yambda-50M as float32 in `data/processed/emb50m_*.npy` |
| `scripts/profile_yambda.py` | day-one profile, writes `outputs/day1_profile.txt` |
| `scripts/peek_data.py` | prints the first rows of each file |
| `notebooks/01_initial_analysis.ipynb` | first organic-vs-recommended comparison |
| `notebooks/02_broaden_or_narrow_three_methods.ipynb` | diversity, MinHash/LSH (own), A-Priori/PCY (own) |
| `notebooks/03_embeddings_sound_analysis.ipynb` | embeddings: audio spread, own spherical k-means pseudo-genres, session coherence, user similarity |
| `data/raw/yambda/` | raw files (50M events, mappings, embeddings) |
| `data/processed/` | extracted 50M embeddings |

Run everything from any folder with `python ...` (`uv` is not installed). Notebooks find the project root themselves.

## 3. Findings on Yambda-50M (10,000 users, 46.5M listens, 301 days)
- `is_organic`: 1 = organic, 0 = recommended (dataset card). Behaviour supports it: 65% of first listens of a (user, track) come via recommendation, 77% of organic listens are repeats.
- Streams are balanced (51.7% organic) and mostly separate in time (only 12.9% of sessions contain both).
- Fair comparison rules used everywhere: real listens only (>= 50% played), distinct tracks (not raw plays), same users in both streams, equal-sized samples, equal basket-size distributions.
- Results (all associations, not causation):
  - **Artists: recommended is broader.** About 11 more distinct artists per 100 tracks, 0.5 bit more entropy, for about 75-80% of users. Robust over 6 configurations.
  - **Sound: recommended is narrower.** Audio spread 0.658 vs 0.698; 13 vs 15 pseudo-genres (k=32); only 6% vs 10% of tracks in genres the user never picks in the other stream. Same sign in 7 configurations incl. popularity controls (the top-20% control keeps only 420 users).
  - **Within a session, recommended is more varied** (coherence 0.417 vs 0.477).
  - **User similarity: mixed.** Mean pairwise similarity slightly lower in the recommended stream, but 75% of users have a closer "twin" there. Frequent itemsets: equal number of frequent pairs, higher typical lift in the recommended stream, more extreme lifts in the organic one.
  - Working interpretation (hypothesis): "broader in artists, narrower in sound".
- Own implementations, each validated: MinHash/LSH vs exact Jaccard (S-curve matches theory, recall 0.87-0.90 with 4% of pairs checked), A-Priori and PCY vs a naive counter (identical; PCY counts about 100x fewer candidates), spherical k-means vs scikit-learn (same mean cosine 0.683 vs 0.682, ARI about 0.7).
- Effect sizes are modest (+17% artists, -6% audio spread, about 2 genres of 15); the tiny p-values come from the sample size.

## 4. Machine limits
15.2 GB RAM (only about 5.5 GB free with other apps open), 16 logical CPUs, about 93 GB free disk, Windows. Close heavy apps before big runs and keep one run under about 8 GB.

## 5. NEXT: scaling to Yambda-500M
Download first: `python scripts/download_yambda_500m.py` (then `--check` is not needed). Everything else (mappings, embeddings) is already local.

Why partitioning is needed: 466M listens would need an estimated 30 GB or more in memory (not measured). All our computations are per user or per session, so we split by user (partitions of about 2,500-10,000 users), compute small additive results per partition and merge (MapReduce/SON idea).

Phase 0: download; extract embeddings for the roughly 3M tracks of 500M; check whether the 50M users are a subset of the 500M users (if yes: the 500M pipeline must reproduce the 50M results for those users, an exact correctness test).
Phase 1: move heavy code from the notebooks into one shared module + scripts; read the parquet one user partition at a time; store per-partition results as Parquet; notebooks only plot.
Phase 2: per method
- Diversity / audio spread: per user; the popularity count (users per track) is additive, collect it in a first pass.
- Frequent itemsets: SON (local threshold, candidate union, one exact counting pass); matching basket sizes across streams needs a global count per size first.
- MinHash/LSH: dense all-pairs matrix is infeasible (about 3.5 billion entries for roughly 60,000 users, estimated, vs about 35 million now); LSH banding scales; compute the exact baseline in blocks to measure recall; likely a more selective (b, r) setting because candidates grow about 100x.
- k-means: fit on a sample, assign all tracks; fits in memory.
Phase 3: scaling experiments: 50M vs 500M, partition sizes, 1-4 workers; record wall time and peak memory per method. Hypotheses: counting methods about linear, exact Jaccard about quadratic, LSH in between.
Phase 4: check that conclusions hold at 500M (effect sizes and confidence intervals), plot scaling curves.

## 6. Open decisions
- Confirm formal requirements with the official brief (distributed framework? report length? outside method?).
- Which 2-3 methods are the formal course methods in the report (currently: MinHash/LSH, A-Priori/PCY, k-means).
- Which research question(s) are core vs stretch (the original summary suggested RQ6, drift over time, as stretch).
- Optional extras not done: graph analysis (Louvain etc., Week 7), ALS as outside method, drift over time.
