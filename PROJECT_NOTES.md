# Project notes: do recommendations broaden or narrow listening? (Yambda)

Last updated: 2026-11-16. Hand-in: 4 December 2026.
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
| `scripts/streams.py` | NEW: `stream_labels()`, the shared stream rule used by all notebooks and the profile (see section 3) |
| `notebooks/01_initial_analysis.ipynb` | first organic-vs-recommended comparison |
| `notebooks/02_broaden_or_narrow_three_methods.ipynb` | diversity, MinHash/LSH (own), A-Priori/PCY (own) |
| `notebooks/03_embeddings_sound_analysis.ipynb` | embeddings: audio spread, own spherical k-means pseudo-genres, session coherence, user similarity |
| `notebooks/04_louvain_sound_graph.ipynb` | NB04: RQ4 Method B (Louvain), one half of RQ4: own Louvain on a kNN graph of audio embeddings (top 50k tracks), tests vs networkx, stability sweep |
| `RQ4_pipeline/` | k-means vs Louvain: notebooks 00-04 (Louvain groups for all tracks, comparison, co-listening graph, listening space, main question with Louvain); see `RQ4_pipeline/README.md` |
| `scripts/louvain.py`, `scripts/rq4_tools.py` | Peter's Louvain as a module; shared helpers for RQ4_pipeline |
| `data/raw/yambda/` | raw files (50M events, mappings, embeddings) |
| `data/processed/` | extracted 50M embeddings |

Run everything from any folder with `python ...` (`uv` is not installed). Notebooks find the project root themselves.

## 3. Findings on Yambda-50M (10,000 users, 46.5M listens, 301 days)
- `is_organic`: 1 = organic, 0 = recommended (dataset card). Behaviour supports it: 65% of first listens of a (user, track) come via recommendation, 77% of organic listens are repeats.
- Streams are balanced (51.7% organic) and mostly separate in time (only 12.9% of sessions contain both).
- Fair comparison rules used everywhere: real listens only (>= 50% played), distinct tracks (not raw plays), same users in both streams, equal-sized samples, equal basket-size distributions.
- **Stream rule (since 2026-11-16):** each (user, track) belongs to one stream only, the stream of its first real listen; tracks heard in the user's first 60 days are excluded (`STREAM_MODE`, `WARMUP_DAYS` in each notebook; `"listen"` restores the old per-listen rule). Excludes 59% of (user, track) pairs; default samples shrink from about 5,500 to about 3,500 users.
- Results (all associations, not causation):
  - **Artists: recommended is broader.** 78 vs 61 distinct artists per 100 tracks, 6.15 vs 5.38 bits entropy, for 78-83% of users. Same direction in 9 configurations (old rule: 78 vs 65).
  - **Sound: recommended is narrower.** Audio spread 0.680 vs 0.727; 14 vs 16 pseudo-genres (k=32); 8% vs 13% of tracks in genres the user never picks in the other stream. Same sign in 10 configurations incl. old rule and popularity controls (the top-20% control keeps only 185 users).
  - **Within a session, recommended is more varied** (coherence 0.418 vs 0.464).
  - **User similarity:** artist sets: users less alike in the recommended stream (mean Jaccard 0.0126 vs 0.0150, similar pairs about equal; under the old rule this was mixed). Audio centroids: less alike on average, but 78% of users have a closer "twin" there. Frequent itemsets: half as many frequent pairs in the recommended stream (458 vs 910) but with higher lift (median 9.1 vs 4.6).
  - Working interpretation (hypothesis): "broader in artists, narrower in sound".
- Own implementations, each validated: MinHash/LSH vs exact Jaccard (S-curve matches theory, recall 0.87-0.90 with 4% of pairs checked), A-Priori and PCY vs a naive counter (identical; PCY counts about 100x fewer candidates), spherical k-means vs scikit-learn (same mean cosine 0.683 vs 0.682, ARI about 0.7).
- Effect sizes are modest (+28% artists, -6% audio spread, about 2 genres of 16); the tiny p-values come from the sample size.
- `reports/*.html` still show the numbers of the old per-listen rule.
- RQ4 Method B (NB04, one half of RQ4): d = 30, k = 20, γ = 1 → 26-28 groups, seed ARI 0.77, artist purity 0.85 (k-means 0.79); ARI vs k-means pseudo-genres 0.43. RQ4 comparison done in `RQ4_pipeline/`: Louvain vs k-means ARI 0.40-0.45 / NMI 0.61-0.63 (k-means self-agreement 0.62-0.69, Louvain 0.80); core tracks agree 74%, border tracks 38%; Louvain groups follow co-listening more (Q 0.58 vs 0.52) but equal without same-artist/album links (0.38 vs 0.37); main question unchanged with Louvain groups (recommended narrower, P 0.43-0.47).

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
