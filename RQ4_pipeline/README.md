# RQ4 pipeline: k-means versus Louvain

RQ4 asks whether our pseudo-genres are real structure in the music or an artefact of k-means. This folder compares the two clustering methods and repeats the cluster-based parts of the `RQ4_extension` analyses with Louvain groups next to the k-means pseudo-genres.

- **Method A:** spherical k-means, k = 32 (`RQ4_kmeans/`)
- **Method B:** Louvain on a kNN graph of the audio embeddings (`notebooks/04_louvain_sound_graph.ipynb`, Peter)

Run the notebooks in order. 00 must run first, because the others read its output.

| Notebook | Question | Mirrors | Run time / peak memory |
|---|---|---|---|
| `00_louvain_groups` | Louvain groups for all 848k tracks (Peter's setting re-run, other tracks assigned by a 20-neighbour vote) | — | ~15 min / 2 GB |
| `01_compare_kmeans_louvain` | How similar are the two partitions, where do they differ, and is either one better? | — | ~3 min / 2 GB |
| `02_partitions_in_listening_graph` | Which partition follows co-listening better? Also Louvain on the co-listening graph as an upper reference | `RQ4_extension/01` | ~5 min / 5 GB |
| `03_partitions_in_listening_space` | Cohesion in listening-embedding space, CCA directions, partitions found from listening, organic vs recommended | `RQ4_extension/02` | ~3 min / 2 GB |
| `04_main_question_with_louvain` | Is recommended listening narrower when genres are Louvain groups? | `RQ4_extension/03` | ~4 min / 5 GB |

Shared code: `scripts/louvain.py` (Peter's implementation, copied unchanged from his notebook) and `scripts/rq4_tools.py` (k-means, partition-comparison helpers, data loading). All notebooks use the shared stream rule in `scripts/streams.py`.

## Main results (Yambda-50M)

| | Result |
|---|---|
| **Agreement** (50k graph tracks) | ARI 0.40–0.45, NMI 0.61–0.63 (chance 0). Two runs of k-means agree at 0.62–0.69 ARI and two runs of Louvain at 0.80, so the methods reach about ⅔ (ARI) to ⅘ (NMI) of the k-means self-agreement. Stable for k = 28–64. |
| **Where they differ** | 9 groups match one to one (30% of tracks). 5 Louvain groups merge pseudo-genres and 9 pseudo-genres are split. The differences run in both directions: Louvain is neither coarser nor finer overall. |
| **Borders** | Core tracks agree 74% of the time, border tracks 38%. This fits a continuum with denser regions, not separate islands. |
| **Criteria** | Each method wins its own criterion but scores well on the other's (Louvain reaches 97% of k-means' centre cosine, k-means 90% of Louvain's modularity). Louvain keeps artists together better (purity 0.84 vs 0.79) and is more reproducible. Giving k-means Louvain's input moves it as much as switching algorithm does. |
| **Co-listening graph** | Louvain groups keep more listening weight inside them: 64% vs 58% (Q 0.58 vs 0.52). Without same-artist/album links the two are equal (0.377 vs 0.371). Sound-only groups reach 61–68% of the modularity of communities found from listening (47% without artist/album links). |
| **Listening space** | Both partitions are cohesive in listening space (lift 0.27 Louvain vs 0.28 k-means) and agree equally with partitions found from listening (NMI ~0.41). Recommended listening is more cohesive than organic under both. |
| **Main question** | Recommended listening is **narrower** with Louvain groups as well: P(rec > org) 0.43–0.47, 12 of 12 comparisons in each of 3 samples. The effect size matches k-means at the same number of groups. It grows with N, and the early-vs-late pattern is the same. |

**Answer to RQ4:** the pseudo-genres are not an artefact of k-means. A graph-based algorithm on different principles recovers their core, finds them in listening behaviour to the same degree, and gives the same answer to the main question. Where group borders fall is partly arbitrary, so statements about single pseudo-genres are fragile and statements about overall diversity are robust.

## Caveats

- **Re-running Louvain does not reproduce Peter's output bit for bit.** Modularity is the same, but cluster counts per seed differ slightly and seed ARI is 0.80 against his 0.77. The most likely cause is floating-point rounding that differs between macOS and Windows. All notebooks here use the groups saved in `louvain_groups.csv.gz`.
- **Outside the top 50,000 tracks, Louvain groups come from a neighbour vote.** It is 97.5% accurate on held-out graph tracks, and the result in 04 is the same on graph tracks only.
- **Graph choice.** Peter's Method B uses a sound graph, while the briefing described a co-listening graph. Notebook 02 runs Louvain on the co-listening graph too, so both versions are covered.

## Output files

`louvain_groups.csv.gz` (groups per track and seed), `compare_*.csv`, `listening_graph_*.csv`, `listening_space_*.csv`, `main_question_*.csv`.
