"""Why did the final candidate set miss each of the true pairs it missed?  (fold 0, artifacts/candidates/fold0)

For every missed pair the script gathers both records (raw text, parsed fields, name representations), measures
  * the rank of the true record in the name + address retrieval when the list is extended to 100 (and its cosine),
  * whether the tokens the two records share were pruned from the retrieval as too common (df > 1%),
  * whether an exact-name or city block containing the pair was dropped by a block cap,
  * name-side signals (script, transliteration, abbreviation, typo, alias / domain forms) and address-side signals,
and assigns one primary cause from the fixed taxonomy: missing name, missing address, country variation, candidate
limit too small, blocking collision, retrieval failure, transliteration, abbreviation, typo, name variation,
address variation, other.  For each cause it then measures which existing or cheap strategies would recover the
pairs (from the coverage flags and from small extra joins), with their global volume, so that the fix chosen for
each cause is the cheapest one that works.
Run from the project root (alone: it needs ~4 GB):  python -u eda/scripts/s16_miss_analysis.py
"""
import gc
import json
import os
import re
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import candidates as C  # noqa: E402

PRE, VAL, OUT_DIR = "artifacts/preprocessed/fold0", "artifacts/validation", "artifacts/candidates/fold0"
OUT_JSON = "eda/results/s16_miss_analysis.json"
N_THREADS = 12
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {}

# --------------------------------------------------------------------------- #
# Tables, positions exactly as in s15
# --------------------------------------------------------------------------- #
S1_COLS = ["entity_id", "country", "name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted", "name_translit_core",
           "name_core_tokens", "name_acronym", "addr_tokens", "addr_sorted", "addr_components", "house_number", "street", "city_norm",
           "state", "postal_code", "addr_digit_runs"]
EXTRA = ["business_name", "business_address", "name_script", "name_flags", "addr_missing", "name_legal_forms", "unit", "locality", "landmark"]
folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold"]).filter(pl.col("fold") == 0)
s1 = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=S1_COLS).join(folds.select("entity_id"), on="entity_id", how="inner")
s1 = C.with_positions(s1, "q_idx")
s1 = s1.join(pl.read_parquet(f"{PRE}/train_source1.parquet", columns=["entity_id", *EXTRA]), on="entity_id", how="left")
POOL_FILES = [f"{PRE}/train_source{k}.parquet" for k in (2, 3)]
cov = pl.read_parquet(os.path.join(OUT_DIR, "coverage.parquet"))
STRATS = [c for c in cov.columns if c not in ("q_idx", "p_idx", "src", "in_final", "in_union")]
miss = cov.filter(~pl.col("in_final"))
misses_saved = pl.read_parquet(os.path.join(OUT_DIR, "misses.parquet"), columns=["source1_entity_id", "q_idx", "p_idx"])
chk = miss.select("q_idx", "p_idx").join(s1.select("q_idx", "entity_id"), on="q_idx").join(misses_saved, on=["q_idx", "p_idx"], how="inner")
assert chk.height == miss.height and (chk["entity_id"] == chk["source1_entity_id"]).all(), "q_idx positions do not match the saved misses"
log(f"missed pairs {miss.height:,} of {cov.height:,} (positions verified)")

# pool rows of the missed pairs (p_idx = position in the concatenated S2, S3 files)
p_need = np.sort(miss["p_idx"].unique().to_numpy())
pool_cols = ["entity_id", "country", "name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted", "name_translit_core", "name_core_tokens",
             "addr_tokens", "house_number", "street", "city_norm", "state", "addr_components", *EXTRA]
parts, offset = [], 0
for f in POOL_FILES:
    n = pl.scan_parquet(f).select(pl.len()).collect().item()
    sel = p_need[(p_need >= offset) & (p_need < offset + n)] - offset
    parts.append(pl.read_parquet(f, columns=pool_cols)[sel.tolist()].with_columns(pl.Series("p_idx", (sel + offset).astype(np.uint32))))
    offset += n
pool = pl.concat(parts)
del parts
gc.collect()
pairs = (miss.join(s1.rename({c: f"{c}_1" for c in s1.columns if c not in ("q_idx",)}), on="q_idx")
             .join(pool.rename({c: f"{c}_2" for c in pool.columns if c != "p_idx"}), on="p_idx"))
log(f"pair table {pairs.shape}")

# --------------------------------------------------------------------------- #
# Block sizes of the exact-name and city keys (were the pairs inside a dropped block?)
# --------------------------------------------------------------------------- #
NAME_REPS = ("name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted")


def pool_key_counts(cols):
    df = pl.concat([pl.read_parquet(f, columns=["country", *cols]) for f in POOL_FILES])
    return df.drop_nulls(cols).group_by(["country", *cols]).agg(pl.len().alias("n"))


for rep in NAME_REPS:
    cnt = pool_key_counts([rep]).rename({rep: f"{rep}_1", "n": f"block_{rep}"})
    pairs = pairs.join(cnt, left_on=["country_1", f"{rep}_1"], right_on=["country", f"{rep}_1"], how="left")
cnt = pool_key_counts(["city_norm", "name_core_tokens"]) if False else None
city_first = pl.concat([pl.read_parquet(f, columns=["country", "city_norm", "name_core_tokens"]) for f in POOL_FILES]).with_columns(
    pl.col("name_core_tokens").list.first().alias("name_first")).drop_nulls(["city_norm", "name_first"]).group_by(["country", "city_norm", "name_first"]).agg(pl.len().alias("block_city_first"))
pairs = pairs.with_columns(pl.col("name_core_tokens_1").list.first().alias("name_first_1")).join(
    city_first.rename({"city_norm": "city_norm_1", "name_first": "name_first_1"}), left_on=["country_1", "city_norm_1", "name_first_1"],
    right_on=["country", "city_norm_1", "name_first_1"], how="left")
del city_first
gc.collect()
log("block sizes attached")

# --------------------------------------------------------------------------- #
# Extended retrieval: rank of the true record within the top 100 of the name + address retrieval, and its cosine
# --------------------------------------------------------------------------- #
class TokenVectorizerDF(C.TokenVectorizer):
    """TokenVectorizer that also keeps the raw document frequency of every token (to detect pruned shared tokens)."""

    def fit(self, tokens):
        n = max(tokens.len(), 1)
        ex = self._explode(tokens)
        self.df_all = ex.group_by("tok").agg(pl.len().alias("df"))
        df = self.df_all.filter((pl.col("df") >= self.min_df) & (pl.col("df") <= self.max_df * n)).sort("tok")
        self.vocab = df.with_columns(pl.arange(0, df.height, dtype=pl.Int64).alias("col"))
        self.n_features = self.vocab.height
        self.idf = (np.log((1 + n) / (1 + self.vocab["df"].to_numpy())) + 1.0).astype(np.float32)
        self.n_docs = n
        return self


def name_addr_tokens(df):
    return df.select(pl.concat_list([pl.col("name_translit_core").str.split(" "), pl.col("addr_tokens")]).alias("t"))["t"]


ranks = np.full(pairs.height, -1, dtype=np.int64)
cosines = np.zeros(pairs.height, dtype=np.float32)
pruned_shared = np.zeros(pairs.height, dtype=np.int64)     # shared tokens with df > 1% (dropped from the retrieval)
kept_shared = np.zeros(pairs.height, dtype=np.int64)       # shared tokens inside the vocabulary
rare_shared = np.zeros(pairs.height, dtype=np.int64)       # shared tokens with df < 2 (unique to the pair: dropped by min_df)
offset = 0
for k, f in zip((2, 3), POOL_FILES):
    n = pl.scan_parquet(f).select(pl.len()).collect().item()
    for country in sorted(pairs["country_1"].unique().to_list()):
        sel = np.flatnonzero((pairs["country_1"].to_numpy() == country) & (pairs["src"].to_numpy() == f"S{k}"))
        if len(sel) == 0:
            continue
        ps = (pl.scan_parquet(f).with_row_index("p_idx", offset=offset).filter(pl.col("country") == country)
                .select("p_idx", "name_translit_core", "addr_tokens").collect().with_columns(pl.col("p_idx").cast(pl.UInt32)))
        vec = TokenVectorizerDF(max_df=0.01).fit(name_addr_tokens(ps))
        P = vec.transform(name_addr_tokens(ps))
        p_pos = ps["p_idx"].to_numpy()
        sub = pairs[sel.tolist()]
        Q = vec.transform(name_addr_tokens(sub.rename({"name_translit_core_1": "name_translit_core", "addr_tokens_1": "addr_tokens"})))
        # cosine of the true pairs
        pos_of = {int(p): i for i, p in enumerate(p_pos)}
        tp = np.array([pos_of[int(p)] for p in sub["p_idx"].to_numpy()])
        cos = np.asarray(Q.multiply(P[tp]).sum(axis=1)).ravel()
        cosines[sel] = cos
        # rank within the top 100 (threshold 0 so that low similarities are ranked too)
        res = C.topk_pairs(np.arange(len(sel), dtype=np.uint32), Q, p_pos, P, k=100, threshold=0.0, n_threads=N_THREADS)
        res = res.join(pl.DataFrame({"q_idx": np.arange(len(sel), dtype=np.uint32), "p_idx": sub["p_idx"].to_numpy()}), on=["q_idx", "p_idx"], how="inner")
        r = np.full(len(sel), -1, dtype=np.int64)
        r[res["q_idx"].to_numpy()] = res["rank"].to_numpy()
        ranks[sel] = r
        # shared tokens: pruned (df > 1%), rare (df < 2) or kept
        df_map = dict(zip(vec.df_all["tok"].to_list(), vec.df_all["df"].to_list()))
        limit = vec.max_df * vec.n_docs
        t1 = name_addr_tokens(sub.rename({"name_translit_core_1": "name_translit_core", "addr_tokens_1": "addr_tokens"})).to_list()
        t2 = name_addr_tokens(sub.rename({"name_translit_core_2": "name_translit_core", "addr_tokens_2": "addr_tokens"})).to_list()
        for j, (a, b) in enumerate(zip(t1, t2)):
            shared = set(a) & set(b)
            pr = sum(1 for t in shared if df_map.get(t, 0) > limit)
            ra = sum(1 for t in shared if df_map.get(t, 0) < 2)
            pruned_shared[sel[j]] = pr
            rare_shared[sel[j]] = ra
            kept_shared[sel[j]] = len(shared) - pr - ra
        log(f"  retrieval diagnosis {country}/S{k}: {len(sel):,} pairs, {ps.height:,} pool, in top-100: {(r > 0).sum():,}")
        del ps, vec, P, Q, res
        gc.collect()
    offset += n
pairs = pairs.with_columns(pl.Series("retr_rank", ranks), pl.Series("retr_cos", cosines), pl.Series("pruned_shared", pruned_shared),
                           pl.Series("kept_shared", kept_shared), pl.Series("rare_shared", rare_shared))

# --------------------------------------------------------------------------- #
# Name and address signals
# --------------------------------------------------------------------------- #
LEGAL = {"inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "llp", "plc"}
INDIC = re.compile(r"[ऀ-෿]")


def initials(tokens):
    return "".join(t[0] for t in tokens if t)


def abbreviation_signal(a_toks, b_toks):
    """True when an unshared token of one side is a prefix (>= 3 chars) or the initialism of tokens of the other side."""
    ua, ub = [t for t in a_toks if t not in b_toks], [t for t in b_toks if t not in a_toks]
    for x, other in ((ua, b_toks), (ub, a_toks)):
        for t in x:
            if len(t) >= 2 and any(o != t and len(o) >= 4 and (o.startswith(t) or t.startswith(o)) and min(len(t), len(o)) >= 3 for o in other):
                return True
            if 2 <= len(t) <= 6 and t.isalpha():
                ini = initials(other)
                if t in ini and len(t) >= 2 and ini != t and len(other) >= 2 and t == initials([o for o in other if o not in LEGAL])[: len(t)]:
                    return True
    return False


rows = pairs.select("name_translit_core_1", "name_translit_core_2", "name_core_1", "name_core_2", "name_compact_1", "name_compact_2",
                    "name_translit_core_sorted_1", "name_translit_core_sorted_2", "name_script_2", "name_script_1", "name_flags_2", "name_flags_1",
                    "business_name_2", "addr_tokens_1", "addr_tokens_2", "addr_missing_1", "addr_missing_2", "house_number_1", "house_number_2",
                    "city_norm_1", "city_norm_2", "state_1", "state_2", "street_1", "street_2", "country_1", "country_2").to_dicts()
sig = {k: [] for k in ("name_ratio", "name_tok_jac", "name_equal_any", "compact_equal", "translit_equal", "abbrev", "typo", "indic_side", "domain_style",
                       "alias_form", "legal_only", "addr_jac", "house_equal", "city_equal", "state_conflict", "addr_shared_n", "name_empty", "addr_missing")}
for r in rows:
    a, b = r["name_translit_core_1"] or "", r["name_translit_core_2"] or ""
    ta, tb = a.split(), b.split()
    sig["name_ratio"].append(fuzz.ratio(a, b))
    sig["name_tok_jac"].append(len(set(ta) & set(tb)) / max(len(set(ta) | set(tb)), 1))
    sig["name_equal_any"].append(any(r[f"{c}_1"] and r[f"{c}_1"] == r[f"{c}_2"] for c in ("name_core", "name_compact", "name_translit_core_sorted")) or (a and a == b))
    sig["compact_equal"].append(bool(r["name_compact_1"]) and r["name_compact_1"] == r["name_compact_2"])
    sig["translit_equal"].append(bool(a) and a == b)
    sig["abbrev"].append(abbreviation_signal(ta, tb))
    sig["typo"].append(a != b and (fuzz.ratio(a, b) >= 80 or Levenshtein.distance(a, b) <= 2))
    sig["indic_side"].append(r["name_script_2"] not in ("latin", None) or r["name_script_1"] not in ("latin", None) or bool(INDIC.search(r["business_name_2"] or "")))
    fl2 = r["name_flags_2"] or []
    sig["domain_style"].append("domain_style" in fl2 or "url_segment" in fl2)
    sig["alias_form"].append("has_alias" in fl2 or bool(re.search(r"[\[\]()]", r["business_name_2"] or "")))
    sig["legal_only"].append((not b.strip()) or all(t in LEGAL for t in tb))
    A, B = set(r["addr_tokens_1"] or []), set(r["addr_tokens_2"] or [])
    sig["addr_jac"].append(len(A & B) / max(len(A | B), 1))
    sig["addr_shared_n"].append(len(A & B))
    sig["house_equal"].append(bool(r["house_number_1"]) and r["house_number_1"] == r["house_number_2"])
    sig["city_equal"].append(bool(r["city_norm_1"]) and r["city_norm_1"] == r["city_norm_2"])
    sig["state_conflict"].append(bool(r["state_1"]) and bool(r["state_2"]) and r["state_1"] != r["state_2"])
    sig["name_empty"].append(not a.strip() or not b.strip())
    sig["addr_missing"].append(bool(r["addr_missing_1"]) or bool(r["addr_missing_2"]))
pairs = pairs.with_columns(*[pl.Series(k, v) for k, v in sig.items()])
pairs = pairs.with_columns(
    pl.any_horizontal([(pl.col(f"{rep}_1") == pl.col(f"{rep}_2")) & (pl.col(f"block_{rep}") > 200) for rep in NAME_REPS]).fill_null(False).alias("name_block_dropped"),
    ((pl.col("city_norm_1") == pl.col("city_norm_2")) & (pl.col("name_first_1") == pl.col("name_core_tokens_2").list.first()) & (pl.col("block_city_first") > 100)).fill_null(False).alias("city_block_dropped"),
)

# --------------------------------------------------------------------------- #
# Primary cause
# --------------------------------------------------------------------------- #
def cause(r):
    if r["country_1"] != r["country_2"]:
        return "country variation"
    if r["name_empty"] or r["legal_only"]:
        return "missing name"
    if r["addr_missing"]:
        return "missing address"
    if 20 < r["retr_rank"] <= 100:
        return "candidate limit too small"
    if r["name_block_dropped"] or r["city_block_dropped"]:
        return "blocking collision"
    similar_name = r["name_equal_any"] or r["name_ratio"] >= 80
    similar_addr = r["addr_jac"] >= 0.3 or r["house_equal"]
    if similar_name and similar_addr:
        return "retrieval failure"
    if not similar_name:
        if r["indic_side"] and not r["translit_equal"]:
            return "transliteration"
        if r["abbrev"] or r["domain_style"]:
            return "abbreviation"
        if r["typo"]:
            return "typo"
        if r["name_tok_jac"] > 0 or r["name_ratio"] >= 50 or r["alias_form"]:
            return "name variation"
        return "other" if not similar_addr else "name variation"
    return "address variation"


pairs = pairs.with_columns(pl.Series("cause", [cause(r) for r in pairs.to_dicts()]))
order = ["missing address", "retrieval failure", "candidate limit too small", "name variation", "typo", "abbreviation", "transliteration", "address variation",
         "blocking collision", "missing name", "country variation", "other"]
by_cause = {c: int((pairs["cause"] == c).sum()) for c in order}
RES["n_missed"] = pairs.height
RES["by_cause"] = by_cause
RES["by_cause_country"] = {c: {k: int(v) for k, v in pairs.filter(pl.col("cause") == c).group_by("country_1").agg(pl.len()).iter_rows()} for c in order}
RES["by_cause_source"] = {c: {k: int(v) for k, v in pairs.filter(pl.col("cause") == c).group_by("src").agg(pl.len()).iter_rows()} for c in order}
log("causes", json.dumps(by_cause))

# secondary signals per cause
def share(col, mask=None):
    sub = pairs if mask is None else pairs.filter(mask)
    return round(float(sub[col].cast(pl.Float64).mean()), 4) if sub.height else None

RES["signals_by_cause"] = {}
for c in order:
    m = pl.col("cause") == c
    sub = pairs.filter(m)
    if sub.height == 0:
        continue
    RES["signals_by_cause"][c] = {
        "n": sub.height, "name_ratio_median": float(np.median(sub["name_ratio"].to_numpy())), "addr_jac_median": float(np.median(sub["addr_jac"].to_numpy())),
        "retr_rank_in_top100": share("retr_rank", m & (pl.col("retr_rank") > 0)) and int((sub["retr_rank"] > 0).sum()),
        "retr_cos_median": float(np.median(sub["retr_cos"].to_numpy())), "pairs_with_pruned_shared_tokens": int((sub["pruned_shared"] > 0).sum()),
        "pairs_with_no_kept_shared_token": int((sub["kept_shared"] == 0).sum()), "indic": int(sub["indic_side"].sum()), "domain_style": int(sub["domain_style"].sum()),
        "alias_form": int(sub["alias_form"].sum()), "house_equal": int(sub["house_equal"].sum()), "city_equal": int(sub["city_equal"].sum()),
        "state_conflict": int(sub["state_conflict"].sum()), "name_block_dropped": int(sub["name_block_dropped"].sum()), "city_block_dropped": int(sub["city_block_dropped"].sum()),
    }
# rank distribution of the misses in the extended retrieval
rk = pairs["retr_rank"].to_numpy()
RES["retrieval_rank_of_misses"] = {"rank_21_30": int(((rk > 20) & (rk <= 30)).sum()), "rank_31_50": int(((rk > 30) & (rk <= 50)).sum()),
                                   "rank_51_100": int(((rk > 50) & (rk <= 100)).sum()), "not_in_top_100": int((rk < 0).sum()),
                                   "cosine_below_0.2_threshold": int((pairs["retr_cos"].to_numpy() < 0.2).sum())}
log("ranks", json.dumps(RES["retrieval_rank_of_misses"]))

# --------------------------------------------------------------------------- #
# What would recover each cause: existing strategies (coverage flags) and cheap new keys
# --------------------------------------------------------------------------- #
recover = {}
for c in order:
    sub = pairs.filter(pl.col("cause") == c)
    if sub.height == 0:
        continue
    recover[c] = {s: int(sub[s].sum()) for s in STRATS if s in sub.columns and int(sub[s].sum()) > 0}
RES["recovered_by_existing_strategy"] = recover

# cheap new keys evaluated on the full fold (recall on the misses and global volume)
positives = cov.select("q_idx", "p_idx", "src")
country_of = s1.select("q_idx", "country")
N_POOL = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in POOL_FILES)
miss_keys = C.pack(pairs["q_idx"].to_numpy(), pairs["p_idx"].to_numpy())


def try_key(name, fn, cols):
    pool_k = C.with_positions(pl.concat([pl.read_parquet(f, columns=["entity_id", "src", "country", *cols]) for f in POOL_FILES]), "p_idx")
    if "name_first" in cols:
        pool_k = pool_k.with_columns(pl.col("name_core_tokens").list.first().alias("name_first"))
    q = s1 if "name_first" not in cols else s1.with_columns(pl.col("name_core_tokens").list.first().alias("name_first"))
    pr = fn(q, pool_k)
    m = C.measure(pr, positives, s1.height, N_POOL, country_of, assume_unique=True)
    hit_miss = int(np.isin(miss_keys, C.pack(pr["q_idx"].to_numpy(), pr["p_idx"].to_numpy()), kind="sort").sum())
    by_c = {c: int(np.isin(miss_keys[(pairs["cause"] == c).to_numpy()], C.pack(pr["q_idx"].to_numpy(), pr["p_idx"].to_numpy()), kind="sort").sum()) for c in order}
    out = {"pairs": m["pairs"], "mean": m["mean"], "p95": m["p95"], "max": m["max"], "recall_alone": m["recall"], "misses_recovered": hit_miss,
           "misses_recovered_by_cause": {k: v for k, v in by_c.items() if v}}
    log(f"  key {name:32s} mean={m['mean']:.1f} p95={m['p95']:.0f} recovers {hit_miss:,} misses")
    C.save_pairs = None
    pr.write_parquet(os.path.join(OUT_DIR, "strategies", f"x_{name}.parquet"))
    del pool_k, pr
    gc.collect()
    return out


RES["cheap_keys"] = {}
RES["cheap_keys"]["name_any_house_number"] = try_key("name_any_house_number", lambda q, p: C.multi_key_block(q, p, [[rep, "house_number"] for rep in NAME_REPS]), [*NAME_REPS, "house_number"])
RES["cheap_keys"]["name_any_city"] = try_key("name_any_city", lambda q, p: C.multi_key_block(q, p, [[rep, "city_norm"] for rep in NAME_REPS], max_block=100), [*NAME_REPS, "city_norm"])
RES["cheap_keys"]["compact_name_state"] = try_key("compact_name_state", lambda q, p: C.key_block(q, p, ["name_compact", "state"]), ["name_compact", "state"])
RES["cheap_keys"]["house_street_city"] = try_key("house_street_city", lambda q, p: C.key_block(q, p, ["house_number", "street", "city_norm"], max_block=200), ["house_number", "street", "city_norm"])

pairs.select("q_idx", "p_idx", "src", "country_1", "cause", "retr_rank", "retr_cos", "pruned_shared", "kept_shared", "rare_shared", "name_ratio", "name_tok_jac",
             "addr_jac", "house_equal", "city_equal", "state_conflict", "name_block_dropped", "city_block_dropped", "indic_side", "domain_style", "alias_form", "abbrev", "typo",
             "business_name_1", "business_address_1", "business_name_2", "business_address_2").write_parquet(os.path.join(OUT_DIR, "miss_causes.parquet"))
RES["examples"] = {c: pairs.filter(pl.col("cause") == c).head(6).select("business_name_1", "business_address_1", "business_name_2", "business_address_2", "retr_rank", "retr_cos",
                                                                          "name_ratio", "addr_jac").to_dicts() for c in order if by_cause[c]}
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
