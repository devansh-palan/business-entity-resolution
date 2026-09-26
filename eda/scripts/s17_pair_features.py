"""Summarise the fold-0 pairwise feature matrix (artifacts/features/fold0): size, groups, null rates, single-feature AUCs.

Run from the project root after features.py:  python eda/scripts/s17_pair_features.py
Writes eda/results/s17_pair_features.json and renders eda/pair_features.md.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
FEAT_DIR = "artifacts/features/fold0"
OUT_JSON = "eda/results/s17_pair_features.json"
OUT_MD = "eda/pair_features.md"
SAMPLE_PER_PART = 25_000          # positives and negatives drawn per part for the AUC sample
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)

meta = json.load(open(os.path.join(FEAT_DIR, "features_meta.json")))
feats = meta["features"]
parts = sorted(f for f in os.listdir(FEAT_DIR) if f.startswith("part_") and f.endswith(".parquet"))
log(f"{len(parts)} parts, {len(feats)} features")

# --------------------------------------------------------------------------- #
# Pass over the parts: counts, null rates, means by label, a stratified sample for the AUCs
# --------------------------------------------------------------------------- #
n_rows = n_pos = 0
nulls = {f: 0 for f in feats}
sum_pos = {f: 0.0 for f in feats}
sum_neg = {f: 0.0 for f in feats}
cnt_pos = {f: 0 for f in feats}
cnt_neg = {f: 0 for f in feats}
by_country = {}
samples = []
bytes_on_disk = 0
rng = np.random.default_rng(0)
for p in parts:
    path = os.path.join(FEAT_DIR, p)
    bytes_on_disk += os.path.getsize(path)
    df = pl.read_parquet(path)
    n_rows += df.height
    lab = df["label"]
    n_pos += int(lab.sum())
    pos, neg = df.filter(lab == 1), df.filter(lab == 0)
    nn = df.select([pl.col(f).null_count().alias(f) for f in feats]).row(0)
    sp = pos.select([pl.col(f).cast(pl.Float64).sum().alias(f) for f in feats]).row(0)
    sn = neg.select([pl.col(f).cast(pl.Float64).sum().alias(f) for f in feats]).row(0)
    cp = pos.select([pl.col(f).count().alias(f) for f in feats]).row(0)
    cn = neg.select([pl.col(f).count().alias(f) for f in feats]).row(0)
    for f, a, b, c, d, e in zip(feats, nn, sp, sn, cp, cn):
        nulls[f] += a
        sum_pos[f] += b or 0.0
        sum_neg[f] += c or 0.0
        cnt_pos[f] += d
        cnt_neg[f] += e
    agg = df.group_by("country_us").agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"))
    for c, n, ps in agg.iter_rows():
        key = "US" if c == 1 else "India"
        by_country.setdefault(key, {"pairs": 0, "positives": 0})
        by_country[key]["pairs"] += n
        by_country[key]["positives"] += ps
    k = min(SAMPLE_PER_PART, pos.height, neg.height)
    samples.append(pl.concat([pos.sample(k, seed=int(rng.integers(1 << 30))), neg.sample(k, seed=int(rng.integers(1 << 30)))]))
    del df, pos, neg
    log(f"  {p}: {n_rows:,} rows so far, {n_pos:,} positives")
sample = pl.concat(samples)
del samples
log(f"sample for AUCs: {sample.height:,} rows ({int(sample['label'].sum()):,} positives)")


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; nulls excluded."""
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


y = sample["label"].to_numpy().astype(np.int64)
rows = []
for f in feats:
    x = sample[f].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    a = auc(x, y)
    rows.append({"feature": f, "group": next((g for g, v in meta["groups"].items() if f in v), "?"), "null_rate": nulls[f] / n_rows,
                 "mean_pos": sum_pos[f] / cnt_pos[f] if cnt_pos[f] else None, "mean_neg": sum_neg[f] / cnt_neg[f] if cnt_neg[f] else None,
                 "auc": a, "auc_oriented": max(a, 1 - a) if a == a else None})
rows.sort(key=lambda r: -(r["auc_oriented"] or 0))
RES = {"n_pairs": n_rows, "n_positives": n_pos, "positive_rate": n_pos / n_rows, "n_features": len(feats), "groups": {g: len(v) for g, v in meta["groups"].items()},
       "bytes_on_disk": bytes_on_disk, "by_country": by_country, "build": {"seconds": meta["seconds"], "stats": meta["stats"],
       "bucket_seconds_mean": float(np.mean([b["seconds"] for b in meta["buckets"]])), "bucket_peak_gb_max": max(b["peak_gb"] or 0 for b in meta["buckets"])},
       "sample_rows": sample.height, "features": rows}
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
log("saved", OUT_JSON)

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
L = []
add = L.append
num = lambda x: f"{x:,}"
add("# The pairwise feature matrix (fold 0)\n")
add(f"`src/features.py` computed {len(feats)} label-free features for every one of the {num(n_rows)} candidate pairs of the fold-0 set "
    f"({num(n_pos)} positives, {100 * n_pos / n_rows:.2f}%), in {meta['n_buckets']} buckets of Source 1 entities written to `artifacts/features/fold0/part_XX.parquet` "
    f"({bytes_on_disk / 1e9:.2f} GB on disk). Every feature is a function of the two records' preprocessed representations, of unlabelled pool statistics "
    "(TF-IDF vocabularies and name / address frequencies fitted on the pool of the same split, no pruning) and of the candidate set itself; the label is joined "
    f"on afterwards. Build: {meta['seconds'] / 60:.0f} minutes on this machine ({meta['stats']['seconds']:.0f} s for the pool statistics, "
    f"{RES['build']['bucket_seconds_mean']:.0f} s per bucket, every bucket in a fresh process peaking at {RES['build']['bucket_peak_gb_max']:.1f} GB).\n")
add("| group | features | what is in it |")
add("|---|---|---|")
DESC = {"name": "exact matches of the six name representations, acronym / variant / containment matches, token Jaccard-overlap-common, digit overlap and conflict, "
                "legal-form agreement, script, Levenshtein distance and similarity, Jaro-Winkler, token-set and partial ratios (rapidfuzz), compact and "
                "transliterated similarities, length and token-count differences, common prefix and suffix, name-flag indicators of both records",
        "address": "exact normalised / sorted address, token Jaccard-overlap-common, component and number overlaps and conflicts, digit signature, house number "
                   "(exact, digits, conflict), street (exact, type, conflict, Jaro-Winkler), unit, city (exact, conflict, Jaro-Winkler), state, postal code, "
                   "country, landmark and locality overlap, length differences, address-flag indicators of both records, character similarity of the whole address",
        "tfidf": "cosine of the name + address token TF-IDF (the retrieval space), of the address tokens, of the name tokens and of the name character 3-grams",
        "context": "log pool frequency (within the country) of each record's name core and normalised address",
        "cross": "name x address, name x city, name x postal, name x country, address x country, identifier (domain-style name) x name and x address, exact name x house "
                 "and x city, name-cosine x address-cosine, min / max / mean of the name and address similarities",
        "entity_relative": "number of candidates of the entity (total and per source), rank and gap to the best of the name + address cosine, the name similarity "
                           "and the address Jaccard, the margin between the entity's best and second-best cosine, how many candidates share the exact name or address",
        "missingness": "missing name, address, house number, street, city, state, postal code, unit on each side", "source": "Source 2 versus Source 3"}
for g, cols in meta["groups"].items():
    add(f"| {g} | {len(cols)} | {DESC.get(g, '')} |")
add("")
add("**Identifiers.** The sources carry no phone, e-mail, website or registration fields, so the identifier block reduces to what the data offers: the pool "
    "record's name being a web domain (`p_domain_style`, `p_url_segment`), and its cross terms with the name and address similarities (`ident_x_name`, "
    "`ident_x_addr`). Postal codes exist in 0.007% of records, so `postal_eq` and `name_x_postal` are almost always null; `country_eq` is constant (candidates "
    "are generated within the country) and `country_us` carries the country instead.\n")
add(f"Pairs by country: " + ", ".join(f"{k} {num(v['pairs'])} pairs with {num(v['positives'])} positives ({100 * v['positives'] / v['pairs']:.2f}%)" for k, v in by_country.items()) + ".\n")
add("## Single-feature discriminative power\n")
add(f"AUC of each feature alone on a stratified sample of {num(sample.height)} pairs (half positives), oriented so that 0.5 is chance and 1 is perfect "
    "(nulls excluded); the null rate and the mean by label are over all pairs. Top 40:\n")
add("| feature | group | AUC | null rate | mean (match) | mean (non-match) |")
add("|---|---|---|---|---|---|")
fmt = lambda v: "-" if v is None else f"{v:.3f}"
for r in rows[:40]:
    add(f"| {r['feature']} | {r['group']} | {fmt(r['auc_oriented'])} | {100 * r['null_rate']:.1f}% | {fmt(r['mean_pos'])} | {fmt(r['mean_neg'])} |")
add("")
weak = [r for r in rows if r["auc_oriented"] is not None and r["auc_oriented"] < 0.55]
add(f"{len(weak)} features have an AUC below 0.55 on their own (" + ", ".join(r["feature"] for r in weak[:25]) + ("..." if len(weak) > 25 else "") + "); "
    "they stay in the matrix because a tree model can use them in interactions and their cost is nil, but none of them will carry the matcher.\n")
best_by_group = {}
for r in rows:
    best_by_group.setdefault(r["group"], r)
add("Best feature per group: " + ", ".join(f"{g} `{r['feature']}` ({fmt(r['auc_oriented'])})" for g, r in best_by_group.items()) + ".\n")
add("## Leakage check\n")
add("No feature reads the label or the truth table: the pool statistics are fitted on the unlabelled pool, the dictionaries used by the preprocessing were learned "
    "without fold 0, and the entity-relative features only look at the candidate set. The label column is added by a join on (q_idx, p_idx) after every feature "
    "is computed, and the same code path runs on the test split, where no label exists.\n")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
log("wrote", OUT_MD)
