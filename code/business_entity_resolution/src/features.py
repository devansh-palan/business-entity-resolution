"""Pairwise feature matrix for candidate pairs (Source 1 entity x Source 2/3 record).

Every feature is a function of the two records' preprocessed representations, of unlabelled pool statistics
(TF-IDF vocabularies and name/address frequencies learned on the pool of the same split) and of the candidate
set itself (how many candidates an entity has, the rank of a pair among them).  Labels are never read while
computing features; when a truth table is given, a ``label`` column is joined on afterwards for training.

The pairs are processed in buckets of Source 1 entities (``q_idx % n_buckets``), so every candidate of an
entity sits in the same bucket (needed for the entity-relative features) and memory stays bounded: every bucket runs
in a fresh child process and is written to ``<out_dir>/part_XX.parquet``.

Usage (from the project root):
    python code/business_entity_resolution/src/features.py --pre-dir artifacts/preprocessed/fold0 --prefix train \
        --pairs artifacts/candidates/fold0/final_pairs.parquet --entity-ids eda/results/_fold0_ids.txt \
        --labels artifacts/candidates/fold0/coverage.parquet --out-dir artifacts/features/fold0 --buckets 32
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates as C  # noqa: E402

SIDE_COLS = ["country", "business_name", "name_clean", "name_core", "name_core_sorted", "name_compact", "name_acronym", "name_translit_core",
             "name_translit_core_sorted", "name_core_tokens", "name_legal_forms", "name_variants", "name_digits", "name_script", "name_flags",
             "addr_norm", "addr_sorted", "house_number", "street", "street_type", "unit", "city_norm", "state", "postal_code", "landmark", "locality",
             "addr_numbers", "addr_digit_runs", "addr_tokens", "addr_components", "addr_missing", "addr_flags"]
NAME_FLAGS_POOL = ["domain_style", "url_segment", "all_caps", "has_alias", "is_acronym", "phone_stripped", "mixed_script", "repeat_removed", "spelling_variant"]
NAME_FLAGS_S1 = ["repeat_removed", "spelling_variant"]
ADDR_FLAGS = ["unit", "reordered", "landmark", "care_of", "indic", "null_token"]
AFFIX_WIDTH = 32
DECOY_FEATURES = os.environ.get("BER_DECOY_FEATURES", "0") == "1"      # decoy-targeted features (house-number edits, digit-stripped equality, number subsets)

GROUPS: Dict[str, List[str]] = {}


def _mem() -> str:
    try:
        import psutil
        m = psutil.Process().memory_info()
        return f"rss {m.rss / 1e9:.2f} GB peak {getattr(m, 'peak_wset', m.rss) / 1e9:.2f} GB"
    except Exception:  # noqa: BLE001
        return ""


def _group(name: str, cols: Iterable[str]) -> None:
    GROUPS.setdefault(name, [])
    for c in cols:
        if c not in GROUPS[name]:
            GROUPS[name].append(c)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_s1(pre_dir: str, prefix: str, entity_ids: Optional[Sequence[str]] = None) -> pl.DataFrame:
    """Source 1 rows with q_idx positions: the same order as generate_candidates.load_queries (file order,
    restricted to ``entity_ids`` when given)."""
    df = pl.read_parquet(os.path.join(pre_dir, f"{prefix}_source1.parquet"), columns=["entity_id", *SIDE_COLS])
    if entity_ids is not None:
        df = df.join(pl.DataFrame({"entity_id": list(entity_ids)}), on="entity_id", how="inner")
        assert df["entity_id"].to_list() == list(entity_ids), "the Source 1 order must be the file order of the id list"
    return C.with_positions(df, "q_idx")


def pool_files(pre_dir: str, prefix: str) -> List[str]:
    return [os.path.join(pre_dir, f"{prefix}_source{k}.parquet") for k in (2, 3)]


def pool_sizes(files: Sequence[str]) -> List[int]:
    return [pl.scan_parquet(f).select(pl.len()).collect().item() for f in files]


def iter_pool_chunks(files: Sequence[str], cols: Sequence[str], chunk: int = 1_000_000) -> Iterable[pl.DataFrame]:
    for f in files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        for i in range(0, n, chunk):
            yield pl.scan_parquet(f).slice(i, chunk).select(cols).collect()


def load_pool_rows(files: Sequence[str], sizes: Sequence[int], p_idx: np.ndarray, cols: Sequence[str]) -> pa.Table:
    """The pool rows at the given global positions, in that order (duplicates allowed), as an Arrow table.

    Row groups are read one at a time with pyarrow on the system memory pool, so the memory of a row group is
    returned to the OS as soon as it is dropped (Polars' allocator keeps freed pages on Windows)."""
    pa.set_memory_pool(pa.system_memory_pool())
    order = np.argsort(p_idx, kind="stable")
    sorted_idx = p_idx[order]
    parts, offset = [], 0
    for f, n in zip(files, sizes):
        pf = pq.ParquetFile(f)
        start = 0
        for rg in range(pf.metadata.num_row_groups):
            nr = pf.metadata.row_group(rg).num_rows
            lo, hi = offset + start, offset + start + nr
            sub = sorted_idx[(sorted_idx >= lo) & (sorted_idx < hi)]
            if len(sub):
                tbl = pf.read_row_group(rg, columns=list(cols))
                parts.append(tbl.take(pa.array(sub - lo)))
                del tbl
            start += nr
        offset += n
    gathered = pa.concat_tables(parts) if parts else pa.table({c: [] for c in cols})
    del parts
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    return gathered.take(pa.array(inverse)).combine_chunks()          # back to the pairs' order


# --------------------------------------------------------------------------- #
# Pool statistics (unsupervised): TF-IDF vocabularies and name / address frequencies
# --------------------------------------------------------------------------- #
def name_addr_tokens(df: pl.DataFrame, suffix: str = "") -> pl.Expr:
    return pl.concat_list([pl.col(f"name_translit_core{suffix}").str.split(" "), pl.col(f"addr_tokens{suffix}")])


TOKEN_KINDS: Dict[str, Tuple[List[str], Callable[[pl.DataFrame, str], pl.Series]]] = {
    "name_addr": (["name_translit_core", "addr_tokens"], lambda d, s: d.select(name_addr_tokens(d, s).alias("t"))["t"]),
    "addr": (["addr_tokens"], lambda d, s: d[f"addr_tokens{s}"]),
    "name": (["name_translit_core"], lambda d, s: d.select(pl.col(f"name_translit_core{s}").str.split(" ").alias("t"))["t"]),
}


def fit_token_vectorizer_streaming(files: Sequence[str], kind: str, chunk: int = 500_000) -> C.TokenVectorizer:
    """TokenVectorizer with the full pool vocabulary (no pruning); document frequencies are accumulated chunk by chunk
    from the unique tokens of every record (no (row, token) hash table)."""
    cols, text_of = TOKEN_KINDS[kind]
    dfs, n = [], 0
    for part in iter_pool_chunks(files, cols, chunk):
        toks = text_of(part, "")
        n += toks.len()
        dfs.append(pl.DataFrame({"tok": toks.list.unique()}).explode("tok").drop_nulls("tok").group_by("tok").agg(pl.len().alias("df")))
        del toks, part
    vec = C.TokenVectorizer(min_df=1, max_df=1.0)
    df = pl.concat(dfs).group_by("tok").agg(pl.col("df").sum()).sort("tok")
    vec.vocab = df.with_columns(pl.arange(0, df.height, dtype=pl.Int64).alias("col"))
    vec.n_features = vec.vocab.height
    vec.idf = (np.log((1 + n) / (1 + vec.vocab["df"].to_numpy())) + 1.0).astype(np.float32)
    return vec


def fit_ngram_vectorizer_streaming(files: Sequence[str], col: str = "name_translit_core", chunk: int = 500_000) -> C.CharNgramVectorizer:
    vec = C.CharNgramVectorizer(min_df=1, max_df=1.0)
    df, n = np.zeros(1 << 24, dtype=np.int64), 0
    for part in iter_pool_chunks(files, [col], chunk):
        texts = part[col].to_list()
        n += len(texts)
        r, c = vec._codes(texts)
        _, cols, _ = vec._unique_pairs(r, c)
        df += np.bincount(cols, minlength=1 << 24)
    keep = df >= 1
    vec.col_of = np.full(1 << 24, -1, dtype=np.int64)
    vec.col_of[keep] = np.arange(int(keep.sum()))
    vec.n_features = int(keep.sum())
    vec.idf = np.log((1 + n) / (1 + df[keep])) + 1.0
    return vec


def _key_hash(country: str, col: str) -> pl.Expr:
    """64-bit hash of (country, value); the same expression is used on the pool and on the pair frame."""
    return pl.struct([pl.col(country).alias("c"), pl.col(col).alias("v")]).hash(seed=0)


def pool_frequencies(files: Sequence[str], chunk: int = 1_000_000) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """How many pool records of a country carry a given name core / normalised address, keyed by the hash of (country, value)."""
    names, addrs = [], []
    for part in iter_pool_chunks(files, ["country", "name_core", "addr_norm"], chunk):
        names.append(part.drop_nulls("name_core").select(_key_hash("country", "name_core").alias("k")).group_by("k").agg(pl.len().alias("n")))
        addrs.append(part.drop_nulls("addr_norm").select(_key_hash("country", "addr_norm").alias("k")).group_by("k").agg(pl.len().alias("n")))
    name_df = pl.concat(names).group_by("k").agg(pl.col("n").sum())
    addr_df = pl.concat(addrs).group_by("k").agg(pl.col("n").sum())
    return name_df, addr_df


class PoolStats:
    """Unsupervised pool statistics: the three token TF-IDF vocabularies, the name 3-gram TF-IDF and the name / address
    frequencies.  Fitted once in a child process (so its transient memory goes back to the OS) and kept on disk."""

    def __init__(self):
        self.vec: Dict[str, object] = {}
        self.name_df: Optional[pl.DataFrame] = None
        self.addr_df: Optional[pl.DataFrame] = None

    @classmethod
    def fit(cls, files: Sequence[str], log: Callable = print) -> "PoolStats":
        self = cls()
        t0 = time.time()
        for k in TOKEN_KINDS:
            self.vec[k] = fit_token_vectorizer_streaming(files, k)
            log(f"  token vocabulary {k}: {self.vec[k].n_features:,}  [{time.time() - t0:.0f}s] {_mem()}")
        self.vec["name_ngram"] = fit_ngram_vectorizer_streaming(files)
        log(f"  name 3-gram features {self.vec['name_ngram'].n_features:,}  [{time.time() - t0:.0f}s] {_mem()}")
        self.name_df, self.addr_df = pool_frequencies(files)
        log(f"  pool frequencies: {self.name_df.height:,} names, {self.addr_df.height:,} addresses  [{time.time() - t0:.0f}s] {_mem()}")
        return self

    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        for k in TOKEN_KINDS:
            self.vec[k].vocab.write_parquet(os.path.join(d, f"vocab_{k}.parquet"))
            np.save(os.path.join(d, f"idf_{k}.npy"), self.vec[k].idf)
        ng = self.vec["name_ngram"]
        np.save(os.path.join(d, "ngram_codes.npy"), np.flatnonzero(ng.col_of >= 0))     # column j <-> the j-th kept code
        np.save(os.path.join(d, "ngram_idf.npy"), ng.idf)
        self.name_df.write_parquet(os.path.join(d, "name_freq.parquet"))
        self.addr_df.write_parquet(os.path.join(d, "addr_freq.parquet"))

    @classmethod
    def load(cls, d: str) -> "PoolStats":
        self = cls()
        for k in TOKEN_KINDS:
            v = C.TokenVectorizer(min_df=1, max_df=1.0)
            v.vocab = pl.read_parquet(os.path.join(d, f"vocab_{k}.parquet"))
            v.idf = np.load(os.path.join(d, f"idf_{k}.npy"))
            v.n_features = v.vocab.height
            self.vec[k] = v
        ng = C.CharNgramVectorizer(min_df=1, max_df=1.0)
        codes = np.load(os.path.join(d, "ngram_codes.npy"))
        ng.col_of = np.full(1 << 24, -1, dtype=np.int64)
        ng.col_of[codes] = np.arange(len(codes))
        ng.n_features = len(codes)
        ng.idf = np.load(os.path.join(d, "ngram_idf.npy"))
        self.vec["name_ngram"] = ng
        self.name_df = pl.read_parquet(os.path.join(d, "name_freq.parquet"))
        self.addr_df = pl.read_parquet(os.path.join(d, "addr_freq.parquet"))
        return self


# --------------------------------------------------------------------------- #
# Feature blocks (all on a frame with *_1 = Source 1 side and *_2 = pool side)
# --------------------------------------------------------------------------- #
def _eq(a: str, b: str) -> pl.Expr:
    return (pl.col(a) == pl.col(b)).cast(pl.Int8)


def _both(a: str, b: str) -> pl.Expr:
    return pl.col(a).is_not_null() & pl.col(b).is_not_null()


def _conflict(a: str, b: str) -> pl.Expr:
    """1 when both sides have a value and they differ, 0 when equal, null when either is missing."""
    return pl.when(_both(a, b)).then((pl.col(a) != pl.col(b)).cast(pl.Int8)).otherwise(None)


def _list_jaccard(a: str, b: str, prefix: str) -> List[pl.Expr]:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.Float32)
    la, lb = pl.col(a).list.len().cast(pl.Float32), pl.col(b).list.len().cast(pl.Float32)
    union = la + lb - inter
    return [pl.when(union > 0).then(inter / union).otherwise(None).alias(f"{prefix}_jaccard"),
            pl.when(pl.min_horizontal(la, lb) > 0).then(inter / pl.min_horizontal(la, lb)).otherwise(None).alias(f"{prefix}_overlap"),
            inter.alias(f"{prefix}_common")]


def _flag(col: str, name: str, alias: str) -> pl.Expr:
    return pl.col(col).list.contains(name).cast(pl.Int8).alias(alias)


def polars_features(pf: pl.DataFrame) -> pl.DataFrame:
    """Exact matches, token overlaps, conflicts, lengths, flags and missingness: everything expressible in Polars."""
    e: List[pl.Expr] = []
    # --- name
    e += [_eq("business_name_1", "business_name_2").alias("name_exact_raw"), _eq("name_clean_1", "name_clean_2").alias("name_exact_norm"),
          _eq("name_core_1", "name_core_2").alias("name_exact_core"), _eq("name_core_sorted_1", "name_core_sorted_2").alias("name_exact_sorted"),
          _eq("name_compact_1", "name_compact_2").alias("name_exact_compact"),
          _eq("name_translit_core_sorted_1", "name_translit_core_sorted_2").alias("name_exact_translit_sorted"),
          pl.when((pl.col("name_acronym_1").str.len_chars() >= 2) & (pl.col("name_acronym_2").str.len_chars() >= 2))
            .then(_eq("name_acronym_1", "name_acronym_2")).otherwise(0).alias("name_acronym_eq"),
          ((pl.col("name_acronym_1") == pl.col("name_compact_2")) | (pl.col("name_acronym_2") == pl.col("name_compact_1"))).fill_null(False).cast(pl.Int8).alias("name_acronym_of_other"),
          (pl.col("name_variants_1").list.contains(pl.col("name_core_2")) | pl.col("name_variants_2").list.contains(pl.col("name_core_1"))).fill_null(False).cast(pl.Int8).alias("name_variant_match"),
          pl.when((pl.col("name_compact_1").str.len_chars() >= 4) & (pl.col("name_compact_2").str.len_chars() >= 4))
            .then(pl.col("name_compact_1").str.contains(pl.col("name_compact_2"), literal=True) | pl.col("name_compact_2").str.contains(pl.col("name_compact_1"), literal=True))
            .otherwise(False).fill_null(False).cast(pl.Int8).alias("name_contains")]
    e += _list_jaccard("name_core_tokens_1", "name_core_tokens_2", "name_tok")
    e += [(pl.col("name_core_tokens_1").list.len().cast(pl.Int32) - pl.col("name_core_tokens_2").list.len().cast(pl.Int32)).abs().alias("name_tok_count_diff"),
          (pl.col("name_core_1").str.len_chars().cast(pl.Int32) - pl.col("name_core_2").str.len_chars().cast(pl.Int32)).abs().alias("name_len_diff"),
          pl.min_horizontal(pl.col("name_core_1").str.len_chars(), pl.col("name_core_2").str.len_chars()).cast(pl.Int32).alias("name_len_min"),
          (pl.col("name_core_tokens_1").list.first() == pl.col("name_core_tokens_2").list.first()).cast(pl.Int8).alias("name_first_tok_eq"),
          (pl.col("name_core_tokens_1").list.last() == pl.col("name_core_tokens_2").list.last()).cast(pl.Int8).alias("name_last_tok_eq")]
    e += _list_jaccard("name_digits_1", "name_digits_2", "name_num")
    e += [((pl.col("name_digits_1").list.len() > 0) & (pl.col("name_digits_2").list.len() > 0) & (pl.col("name_digits_1").list.set_intersection(pl.col("name_digits_2")).list.len() == 0))
            .cast(pl.Int8).alias("name_num_conflict"),
          ((pl.col("name_digits_1").list.len() > 0).cast(pl.Int8) + (pl.col("name_digits_2").list.len() > 0).cast(pl.Int8)).alias("name_num_sides"),
          pl.when((pl.col("name_legal_forms_1").list.len() > 0) & (pl.col("name_legal_forms_2").list.len() > 0))
            .then((pl.col("name_legal_forms_1").list.set_intersection(pl.col("name_legal_forms_2")).list.len() == 0).cast(pl.Int8)).otherwise(None).alias("legal_conflict"),
          (pl.col("name_legal_forms_1").list.len() > 0).cast(pl.Int8).alias("legal_1"), (pl.col("name_legal_forms_2").list.len() > 0).cast(pl.Int8).alias("legal_2"),
          _eq("name_script_1", "name_script_2").alias("script_eq"), (pl.col("name_script_2") != "latin").cast(pl.Int8).alias("script_2_nonlatin")]
    e += [_flag("name_flags_2", f, f"p_{f}") for f in NAME_FLAGS_POOL] + [_flag("name_flags_1", f, f"q_{f}") for f in NAME_FLAGS_S1]
    _group("name", [x.meta.output_name() for x in e])
    # --- address
    a: List[pl.Expr] = [_eq("addr_norm_1", "addr_norm_2").alias("addr_exact_norm"), _eq("addr_sorted_1", "addr_sorted_2").alias("addr_exact_sorted")]
    a += _list_jaccard("addr_tokens_1", "addr_tokens_2", "addr_tok")
    a += _list_jaccard("addr_components_1", "addr_components_2", "addr_comp")
    a += _list_jaccard("addr_numbers_1", "addr_numbers_2", "addr_num")
    a += [((pl.col("addr_numbers_1").list.len() > 0) & (pl.col("addr_numbers_2").list.len() > 0) & (pl.col("addr_numbers_1").list.set_intersection(pl.col("addr_numbers_2")).list.len() == 0))
            .cast(pl.Int8).alias("addr_num_conflict"),
          pl.when((pl.col("addr_digit_runs_1").list.len() > 0) & (pl.col("addr_digit_runs_2").list.len() > 0))
            .then((pl.col("addr_digit_runs_1").list.sort().list.join(" ") == pl.col("addr_digit_runs_2").list.sort().list.join(" ")).cast(pl.Int8)).otherwise(None).alias("addr_numsig_eq"),
          _eq("house_number_1", "house_number_2").alias("house_eq"), _conflict("house_number_1", "house_number_2").alias("house_conflict"),
          _eq("house_digits_1", "house_digits_2").alias("house_digits_eq"),
          _eq("street_1", "street_2").alias("street_eq"), _conflict("street_1", "street_2").alias("street_conflict"), _eq("street_type_1", "street_type_2").alias("street_type_eq"),
          _eq("unit_1", "unit_2").alias("unit_eq"), _conflict("unit_1", "unit_2").alias("unit_conflict"),
          _eq("city_norm_1", "city_norm_2").alias("city_eq"), _conflict("city_norm_1", "city_norm_2").alias("city_conflict"),
          _eq("state_1", "state_2").alias("state_eq"), _conflict("state_1", "state_2").alias("state_conflict"),
          _eq("postal_code_1", "postal_code_2").alias("postal_eq"), _eq("country_1", "country_2").alias("country_eq"), (pl.col("country_1") == "US").cast(pl.Int8).alias("country_us")]
    a += _list_jaccard("landmark_1", "landmark_2", "landmark")[:1] + _list_jaccard("locality_1", "locality_2", "locality")[:1]
    a += [(pl.col("addr_norm_1").str.len_chars().cast(pl.Int32) - pl.col("addr_norm_2").str.len_chars().cast(pl.Int32)).abs().alias("addr_len_diff"),
          (pl.col("addr_tokens_1").list.len().cast(pl.Int32) - pl.col("addr_tokens_2").list.len().cast(pl.Int32)).abs().alias("addr_tok_count_diff"),
          pl.min_horizontal(pl.col("addr_tokens_1").list.len(), pl.col("addr_tokens_2").list.len()).cast(pl.Int32).alias("addr_tok_count_min")]
    a += [_flag("addr_flags_2", f, f"p_addr_{f}") for f in ADDR_FLAGS] + [_flag("addr_flags_1", f, f"q_addr_{f}") for f in ADDR_FLAGS]
    if DECOY_FEATURES:
        a += decoy_expressions()
    _group("address", [x.meta.output_name() for x in a])
    # --- missingness and source
    m: List[pl.Expr] = []
    for side in ("1", "2"):
        m += [pl.col(f"name_core_{side}").is_null().cast(pl.Int8).alias(f"name_missing_{side}"), pl.col(f"addr_missing_{side}").cast(pl.Int8).alias(f"addr_missing_{side}"),
              pl.col(f"house_number_{side}").is_null().cast(pl.Int8).alias(f"house_missing_{side}"), pl.col(f"street_{side}").is_null().cast(pl.Int8).alias(f"street_missing_{side}"),
              pl.col(f"city_norm_{side}").is_null().cast(pl.Int8).alias(f"city_missing_{side}"), pl.col(f"state_{side}").is_null().cast(pl.Int8).alias(f"state_missing_{side}"),
              pl.col(f"postal_code_{side}").is_null().cast(pl.Int8).alias(f"postal_missing_{side}"), pl.col(f"unit_{side}").is_null().cast(pl.Int8).alias(f"unit_missing_{side}")]
    _group("missingness", [x.meta.output_name() for x in m])
    s = [pl.col("src_s3")]
    _group("source", ["src_s3"])
    out = pf.with_columns(pl.col("house_number_1").str.extract(r"(\d+)", 1).alias("house_digits_1"), pl.col("house_number_2").str.extract(r"(\d+)", 1).alias("house_digits_2"))
    return out.select(["q_idx", "p_idx", *e, *a, *m, *s])


def decoy_expressions() -> List[pl.Expr]:
    """Decoy signals: the same record with one number edited, dropped or added (on the *_1 / *_2 pair columns)."""
    both_house = _both("house_digits_1", "house_digits_2")
    h1, h2 = pl.col("house_digits_1"), pl.col("house_digits_2")
    nums_1, nums_2 = pl.col("addr_numbers_1"), pl.col("addr_numbers_2")
    both_nums = (nums_1.list.len() > 0) & (nums_2.list.len() > 0)
    strip_digits = lambda c: pl.col(c).str.replace_all(r"\d", "")
    return [pl.when(both_house).then(((h1 != h2) & (h1.str.ends_with(h2) | h2.str.ends_with(h1) | h1.str.starts_with(h2) | h2.str.starts_with(h1))).cast(pl.Int8)).otherwise(None).alias("house_digit_dropped"),
            pl.when(both_house).then((h1.str.len_chars().cast(pl.Int32) - h2.str.len_chars().cast(pl.Int32)).abs()).otherwise(None).alias("house_len_diff"),
            ((strip_digits("addr_norm_1") == strip_digits("addr_norm_2")) & (pl.col("addr_norm_1") != pl.col("addr_norm_2"))).cast(pl.Int8).alias("addr_same_ignoring_numbers"),
            ((strip_digits("name_core_1") == strip_digits("name_core_2")) & (pl.col("name_core_1") != pl.col("name_core_2"))).cast(pl.Int8).alias("name_same_ignoring_numbers"),
            pl.when(both_nums).then((nums_2.list.set_difference(nums_1).list.len() == 0).cast(pl.Int8)).otherwise(None).alias("addr_num_2_subset_of_1"),
            pl.when(both_nums).then((nums_1.list.set_difference(nums_2).list.len() == 0).cast(pl.Int8)).otherwise(None).alias("addr_num_1_subset_of_2"),
            nums_1.list.set_symmetric_difference(nums_2).list.len().cast(pl.Int16).alias("addr_num_sym_diff"),
            ((pl.col("name_core_1") == pl.col("name_core_2")) & (pl.col("street_1") == pl.col("street_2")) & both_house & (h1 != h2)).fill_null(False).cast(pl.Int8).alias("same_name_street_diff_house"),
            ((pl.col("name_core_1") == pl.col("name_core_2")) & (strip_digits("addr_norm_1") == strip_digits("addr_norm_2")) & (pl.col("addr_norm_1") != pl.col("addr_norm_2"))).fill_null(False).cast(pl.Int8).alias("same_name_addr_diff_numbers"),
            (pl.col("unit_1").is_null() != pl.col("unit_2").is_null()).cast(pl.Int8).alias("unit_one_side")]


def _strings(pf: pl.DataFrame, col: str) -> Tuple[List[str], np.ndarray]:
    s = pf[col]
    return s.fill_null("").to_list(), s.is_null().to_numpy()


def _cpdist(a: List[str], b: List[str], scorer: Callable, workers: int, chunk: int = 200_000) -> np.ndarray:
    """Pairwise rapidfuzz scores in slices, so its internal string buffers stay small."""
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = process.cpdist(a[i:i + chunk], b[i:i + chunk], scorer=scorer, workers=workers)
    return out


def string_metric_features(pf: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """rapidfuzz metrics computed pairwise in C++; null on either side gives a null feature."""
    out: Dict[str, np.ndarray] = {}

    def run(col: str, specs: Sequence[Tuple[str, Callable, float]]):
        a, na = _strings(pf, f"{col}_1")
        b, nb = _strings(pf, f"{col}_2")
        miss = na | nb
        for name, scorer, scale in specs:
            v = _cpdist(a, b, scorer, workers)
            if scale != 1.0:
                v /= scale
            v[miss] = np.nan
            out[name] = v
        del a, b

    run("name_core", [("name_lev_dist", Levenshtein.distance, 1.0), ("name_lev_sim", Levenshtein.normalized_similarity, 1.0), ("name_jw", JaroWinkler.similarity, 1.0),
                      ("name_token_set", fuzz.token_set_ratio, 100.0), ("name_partial", fuzz.partial_ratio, 100.0)])
    run("name_compact", [("name_compact_lev_sim", Levenshtein.normalized_similarity, 1.0)])
    run("name_translit_core", [("name_translit_jw", JaroWinkler.similarity, 1.0), ("name_translit_lev_sim", Levenshtein.normalized_similarity, 1.0)])
    run("addr_norm", [("addr_lev_sim", Levenshtein.normalized_similarity, 1.0), ("addr_partial", fuzz.partial_ratio, 100.0), ("addr_token_set", fuzz.token_set_ratio, 100.0)])
    run("street", [("street_jw", JaroWinkler.similarity, 1.0)])
    run("city_norm", [("city_jw", JaroWinkler.similarity, 1.0)])
    if DECOY_FEATURES:
        run("addr_norm", [("addr_lev_dist", Levenshtein.distance, 1.0)])
        run("house_number", [("house_lev_dist", Levenshtein.distance, 1.0)])      # decoys: one digit edited, dropped or added
        run("unit", [("unit_lev_dist", Levenshtein.distance, 1.0)])
    _group("name", ["name_lev_dist", "name_lev_sim", "name_jw", "name_token_set", "name_partial", "name_compact_lev_sim", "name_translit_jw", "name_translit_lev_sim"])
    _group("address", [c for c in ("addr_lev_sim", "addr_partial", "addr_token_set", "street_jw", "city_jw", "addr_lev_dist", "house_lev_dist", "unit_lev_dist") if c in out])
    return pl.DataFrame(out)


def affix_features(pf: pl.DataFrame, col: str = "name_core", width: int = AFFIX_WIDTH) -> pl.DataFrame:
    """Common prefix and suffix length of the name cores (characters, capped at ``width``), raw and relative to the shorter name."""
    def codepoints(s: pl.Series, reverse: bool) -> np.ndarray:
        s = s.fill_null("")
        if reverse:
            s = s.str.reverse()
        arr = np.array(s.str.slice(0, width).to_list(), dtype=f"U{width}")
        return arr.view(np.uint32).reshape(len(arr), width)

    out = {}
    lens = pl.min_horizontal(pl.col(f"{col}_1").str.len_chars(), pl.col(f"{col}_2").str.len_chars())
    min_len = pf.select(lens.alias("m"))["m"].to_numpy().astype(np.float32)
    miss = (pf[f"{col}_1"].is_null() | pf[f"{col}_2"].is_null()).to_numpy()
    for kind, rev in (("prefix", False), ("suffix", True)):
        a, b = codepoints(pf[f"{col}_1"], rev), codepoints(pf[f"{col}_2"], rev)
        eq = (a == b) & (a != 0)
        n = np.cumprod(eq, axis=1).sum(axis=1).astype(np.float32)
        n[miss] = np.nan
        out[f"name_common_{kind}"] = n
        rel = np.where(min_len > 0, n / np.minimum(np.maximum(min_len, 1), width), np.nan).astype(np.float32)
        out[f"name_common_{kind}_rel"] = rel
        del a, b, eq
    _group("name", list(out))
    return pl.DataFrame(out)


def _rowwise_cosine(Q: sp.csr_matrix, P: sp.csr_matrix) -> np.ndarray:
    return np.asarray(Q.multiply(P).sum(axis=1)).ravel().astype(np.float32)


def _sliced_cosine(a, b, vec, n: int, chunk: int = 250_000) -> np.ndarray:
    """Row-wise cosine between the TF-IDF rows of two aligned text sequences, transformed one slice at a time."""
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        out[i:i + chunk] = _rowwise_cosine(vec.transform(a[i:i + chunk]), vec.transform(b[i:i + chunk]))
    return out


def tfidf_features(pf: pl.DataFrame, stats: PoolStats) -> pl.DataFrame:
    out = {}
    for kind in ("name_addr", "addr", "name"):
        cols, text_of = TOKEN_KINDS[kind]
        out[f"cos_{kind}"] = _sliced_cosine(text_of(pf, "_1"), text_of(pf, "_2"), stats.vec[kind], pf.height)
    out["cos_name_ngram"] = _sliced_cosine(pf["name_translit_core_1"].to_list(), pf["name_translit_core_2"].to_list(), stats.vec["name_ngram"], pf.height)
    _group("tfidf", list(out))
    return pl.DataFrame(out)


def context_features(pf: pl.DataFrame, stats: PoolStats) -> pl.DataFrame:
    """Pool frequencies of the two names and the two addresses within the country (log1p), from unlabelled pool counts."""
    out = pf.select(_key_hash("country_1", "name_core_1").alias("kn1"), _key_hash("country_1", "name_core_2").alias("kn2"),
                    _key_hash("country_1", "addr_norm_1").alias("ka1"), _key_hash("country_1", "addr_norm_2").alias("ka2"),
                    pl.col("name_core_1").is_null().alias("nn1"), pl.col("name_core_2").is_null().alias("nn2"),
                    pl.col("addr_norm_1").is_null().alias("na1"), pl.col("addr_norm_2").is_null().alias("na2"))
    for key, tbl, name, nul in (("kn1", stats.name_df, "name_pool_freq_1", "nn1"), ("kn2", stats.name_df, "name_pool_freq_2", "nn2"),
                                ("ka1", stats.addr_df, "addr_pool_freq_1", "na1"), ("ka2", stats.addr_df, "addr_pool_freq_2", "na2")):
        out = out.join(tbl.rename({"k": key, "n": name}), on=key, how="left").with_columns(pl.when(pl.col(nul)).then(0).otherwise(pl.col(name).fill_null(0)).alias(name))
    cols = ["name_pool_freq_1", "name_pool_freq_2", "addr_pool_freq_1", "addr_pool_freq_2"]
    _group("context", cols)
    return out.select([pl.col(c).cast(pl.Float32).log1p().alias(c) for c in cols])


def cross_features(df: pl.DataFrame) -> pl.DataFrame:
    c = [(pl.col("name_lev_sim") * pl.col("addr_tok_jaccard")).alias("name_x_addr"), (pl.col("name_lev_sim") * pl.col("city_eq")).alias("name_x_city"),
         (pl.col("name_lev_sim") * pl.col("postal_eq")).alias("name_x_postal"), (pl.col("name_lev_sim") * pl.col("country_us")).alias("name_x_country"),
         (pl.col("addr_tok_jaccard") * pl.col("country_us")).alias("addr_x_country"), (pl.col("p_domain_style") * pl.col("name_partial")).alias("ident_x_name"),
         (pl.col("p_domain_style") * pl.col("addr_tok_jaccard")).alias("ident_x_addr"), (pl.col("name_exact_core") * pl.col("house_eq")).alias("exact_name_x_house"),
         (pl.col("name_exact_core") * pl.col("city_eq")).alias("exact_name_x_city"), (pl.col("cos_name") * pl.col("cos_addr")).alias("cos_name_x_addr"),
         pl.min_horizontal("name_lev_sim", "addr_lev_sim").alias("sim_min"), pl.max_horizontal("name_lev_sim", "addr_lev_sim").alias("sim_max"),
         pl.mean_horizontal("name_lev_sim", "addr_lev_sim").alias("sim_mean")]
    _group("cross", [x.meta.output_name() for x in c])
    return df.with_columns(c)


def entity_relative_features(df: pl.DataFrame) -> pl.DataFrame:
    """Where the pair stands among the candidates of its Source 1 entity (all candidates of an entity are in the frame)."""
    e = []
    for col, tag in (("cos_name_addr", "cos"), ("name_lev_sim", "name"), ("addr_tok_jaccard", "addr")):
        v = pl.col(col).fill_nan(0.0).fill_null(0.0)
        e += [v.rank(method="min", descending=True).over("q_idx").cast(pl.Int16).alias(f"rank_{tag}_q"),
              v.rank(method="min", descending=True).over(["q_idx", "src_s3"]).cast(pl.Int16).alias(f"rank_{tag}_qs"),
              (v.max().over("q_idx") - v).cast(pl.Float32).alias(f"gap_{tag}_q"), (v.max().over(["q_idx", "src_s3"]) - v).cast(pl.Float32).alias(f"gap_{tag}_qs")]
    v = pl.col("cos_name_addr").fill_nan(0.0).fill_null(0.0)
    second = pl.when(pl.len().over("q_idx") > 1).then(v.top_k(2).min().over("q_idx")).otherwise(0.0)
    e += [pl.len().over("q_idx").cast(pl.Int16).alias("n_cand_q"), pl.len().over(["q_idx", "src_s3"]).cast(pl.Int16).alias("n_cand_qs"),
          (v.max().over("q_idx") - second).cast(pl.Float32).alias("cos_margin_q"), pl.col("name_exact_core").sum().over("q_idx").cast(pl.Int16).alias("n_exact_core_q"),
          pl.col("addr_exact_norm").sum().over("q_idx").cast(pl.Int16).alias("n_exact_addr_q"), (pl.col("cos_name_addr").fill_nan(0.0).fill_null(0.0) == v.max().over("q_idx")).cast(pl.Int8).alias("is_top_cos_q")]
    if DECOY_FEATURES:      # decoy context: candidates that look like the same record with a number changed, and whether a clean twin exists
        e += [pl.col("same_name_street_diff_house").sum().over("q_idx").cast(pl.Int16).alias("n_same_name_street_diff_house_q"),
              pl.col("same_name_addr_diff_numbers").sum().over("q_idx").cast(pl.Int16).alias("n_same_name_addr_diff_numbers_q"),
              ((pl.col("name_exact_core") == 1) & (pl.col("addr_exact_norm") == 1)).cast(pl.Int8).max().over("q_idx").alias("has_exact_twin_q"),
              ((pl.col("name_exact_core") == 1) & (pl.col("house_eq") == 1)).cast(pl.Int8).max().over("q_idx").alias("has_name_house_twin_q")]
    _group("entity_relative", [x.meta.output_name() for x in e])
    return df.with_columns(e)


# --------------------------------------------------------------------------- #
# Bucket driver
# --------------------------------------------------------------------------- #
def pair_frame(pairs: pl.DataFrame, s1: pl.DataFrame, pool_rows: pa.Table, n_s2: int, cols: Sequence[str]) -> pl.DataFrame:
    """One row per pair with the Source 1 columns (*_1) and the pool columns (*_2) for the requested columns only;
    ``pool_rows`` is already in pair order."""
    q = pairs["q_idx"].to_numpy()
    left = s1.select(list(cols))[q].rename({c: f"{c}_1" for c in cols})
    right = pl.from_arrow(pool_rows.select(list(cols))).rename({c: f"{c}_2" for c in cols})
    return pl.concat([pairs.select("q_idx", "p_idx"), left, right], how="horizontal").with_columns((pl.col("p_idx") >= n_s2).cast(pl.Int8).alias("src_s3"))


STRING_COLS = ["country", "name_core", "name_compact", "name_translit_core", "addr_norm", "street", "city_norm", "addr_tokens"]
DECOY_STRING_COLS = ["house_number", "unit"]


def slim(pf: pl.DataFrame) -> pl.DataFrame:
    """Only the columns the string, TF-IDF and context features need (the list-heavy columns are dropped after polars_features)."""
    cols = STRING_COLS + (DECOY_STRING_COLS if DECOY_FEATURES else [])
    return pf.select(["q_idx", "p_idx", *[f"{c}_{s}" for c in cols for s in ("1", "2")]])


def metric_features(base: pl.DataFrame, pf: pl.DataFrame, stats: PoolStats, workers: int = -1, log: Optional[Callable] = None,
                    entity_relative: bool = True) -> pl.DataFrame:
    """Everything after polars_features: ``pf`` is the slimmed pair frame, ``base`` the Polars feature block.  The
    entity-relative block needs every candidate of an entity in the frame, so it is skipped for slices of a bucket."""
    step = (lambda name: log(f"      {name:18s} {_mem()}")) if log else (lambda name: None)
    parts = [base]
    for name, fn in (("string metrics", lambda: string_metric_features(pf, workers)), ("affixes", lambda: affix_features(pf)),
                     ("tfidf", lambda: tfidf_features(pf, stats)), ("context", lambda: context_features(pf, stats))):
        parts.append(fn())
        step(name)
    df = cross_features(pl.concat(parts, how="horizontal"))
    del parts
    if entity_relative:
        df = entity_relative_features(df)
    step("cross + entity")
    float_cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    return df.with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in float_cols])      # one missing-value convention: null


def bucket_features(pf: pl.DataFrame, stats: PoolStats, workers: int = -1) -> pl.DataFrame:
    return metric_features(polars_features(pf), slim(pf), stats, workers)


def feature_columns(df: pl.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("q_idx", "p_idx", "label", "entity_id", "candidate_id")]


def _peak_gb() -> Optional[float]:
    try:
        import psutil
        m = psutil.Process().memory_info()
        return round(getattr(m, "peak_wset", m.rss) / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def _stats_worker(files: Sequence[str], stats_dir: str, pre_dir: str, prefix: str, entity_ids: Optional[Sequence[str]]) -> dict:
    t0 = time.time()
    os.makedirs(stats_dir, exist_ok=True)
    s1 = load_s1(pre_dir, prefix, entity_ids)                     # the Source 1 subset is written once; the bucket workers read it
    s1.write_parquet(os.path.join(stats_dir, "s1.parquet"))
    n_s1 = s1.height
    del s1
    gc.collect()
    st = PoolStats.fit(files, log=lambda *a: print(f"[stats {time.time() - t0:4.0f}s]", *a, flush=True))
    st.save(stats_dir)
    return {"seconds": round(time.time() - t0, 1), "n_entities": int(n_s1), "n_features": {k: int(v.n_features) for k, v in st.vec.items()}, "peak_gb": _peak_gb()}


def _bucket_worker(pre_dir: str, prefix: str, pairs_path: str, labels_path: Optional[str], stats_dir: str, out_dir: str,
                   b: int, n_buckets: int, workers: int, slice_rows: int = 250_000, model: Optional[dict] = None) -> dict:
    """One bucket end to end in its own process: the pool rows of the bucket are gathered once, then the pairs are
    processed in slices of ``slice_rows`` (small transient frames), and the entity-relative features are added on the
    whole bucket.  Returns the bucket's statistics.  With ``model`` (path, features, neg_rate) the bucket is scored
    instead of stored: only q_idx, p_idx and the prior-corrected probability are written (inference mode)."""
    t0 = time.time()
    files = pool_files(pre_dir, prefix)
    sizes = pool_sizes(files)
    s1 = pl.read_parquet(os.path.join(stats_dir, "s1.parquet"))
    pairs = (pl.read_parquet(pairs_path, columns=["q_idx", "p_idx"]).with_columns(pl.col("q_idx").cast(pl.UInt32), pl.col("p_idx").cast(pl.UInt32))
               .filter(pl.col("q_idx") % n_buckets == b).sort(["q_idx", "p_idx"]))
    stats = PoolStats.load(stats_dir)
    rows = load_pool_rows(files, sizes, pairs["p_idx"].to_numpy(), SIDE_COLS)
    outs = []
    for i in range(0, pairs.height, slice_rows):
        pb = pairs.slice(i, slice_rows)
        pf = pair_frame(pb, s1, rows.slice(i, slice_rows), sizes[0], SIDE_COLS)
        base = polars_features(pf)
        pf = slim(pf)
        outs.append(metric_features(base, pf, stats, workers, entity_relative=False))
        del pf, base
        gc.collect()
    del rows, s1, stats
    gc.collect()
    df = entity_relative_features(pl.concat(outs))
    del outs
    float_cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    df = df.with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in float_cols])
    positives = None
    if labels_path:
        lab = pl.read_parquet(labels_path, columns=["q_idx", "p_idx"])
        pos_keys = np.unique(C.pack(lab["q_idx"].to_numpy(), lab["p_idx"].to_numpy()))
        df = df.with_columns(pl.Series("label", np.isin(C.pack(df["q_idx"].to_numpy(), df["p_idx"].to_numpy()), pos_keys, kind="sort").astype(np.int8)))
        positives = int(df["label"].sum())
    cols = feature_columns(df)
    if model is not None:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model["path"])
        feats = model["features"]
        X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy().astype(np.float32, copy=False)
        p = booster.predict(X).astype(np.float64)
        del X
        p = np.clip(p, 1e-7, 1 - 1e-7)
        odds = p / (1 - p) * model["neg_rate"]
        out = df.select("q_idx", "p_idx").with_columns(pl.Series("p", (odds / (1 + odds)).astype(np.float32)))
        out.write_parquet(os.path.join(out_dir, f"pred_{b:02d}.parquet"))
        return {"bucket": b, "pairs": int(df.height), "positives": positives, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(),
                "features": cols, "groups": {g: [c for c in v if c in cols] for g, v in GROUPS.items()}, "scored": True, "mean_p": round(float(out["p"].mean()), 5)}
    df.write_parquet(os.path.join(out_dir, f"part_{b:02d}.parquet"))
    return {"bucket": b, "pairs": int(df.height), "positives": positives, "seconds": round(time.time() - t0, 1), "peak_gb": _peak_gb(),
            "features": cols, "groups": {g: [c for c in v if c in cols] for g, v in GROUPS.items()}}


def build(pre_dir: str, prefix: str, pairs_path: str, out_dir: str, entity_ids: Optional[Sequence[str]] = None, labels_path: Optional[str] = None,
          n_buckets: int = 32, workers: int = -1, log: Callable = print, only_bucket: Optional[int] = None, n_procs: int = 1, in_process: bool = False,
          model: Optional[dict] = None) -> dict:
    """Fit the pool statistics, then compute every bucket; each step runs in a fresh child process (``in_process=False``)
    so the parent stays thin and the memory of a bucket is returned to the OS when it ends.  ``n_procs`` buckets may run
    concurrently on machines with enough RAM."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    files = pool_files(pre_dir, prefix)
    stats_dir = os.path.join(out_dir, "_pool_stats")
    n_pairs = pl.scan_parquet(pairs_path).select(pl.len()).collect().item()
    buckets = [b for b in range(n_buckets) if only_bucket is None or b == only_bucket]
    log(f"{n_pairs:,} pairs; pool {sum(pool_sizes(files)):,} records; {len(buckets)} bucket(s) of {n_buckets}, {n_procs} worker process(es)")
    ids = list(entity_ids) if entity_ids is not None else None
    args = [(pre_dir, prefix, pairs_path, labels_path, stats_dir, out_dir, b, n_buckets, workers, 250_000, model) for b in buckets]
    results = []
    if in_process:
        stats_info = _stats_worker(files, stats_dir, pre_dir, prefix, ids)
        log(f"  pool statistics fitted in {stats_info['seconds']:.0f}s")
        for a in args:
            results.append(_bucket_worker(*a))
            log(f"  bucket {results[-1]['bucket']:2d}: {results[-1]['pairs']:,} pairs  [{results[-1]['seconds']:.0f}s]")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
            stats_info = ex.submit(_stats_worker, files, stats_dir, pre_dir, prefix, ids).result()
        log(f"  pool statistics fitted in {stats_info['seconds']:.0f}s (child peak {stats_info['peak_gb']} GB): " + ", ".join(f"{k} {v:,}" for k, v in stats_info["n_features"].items()))
        with ProcessPoolExecutor(max_workers=n_procs, max_tasks_per_child=1) as ex:
            futures = [ex.submit(_bucket_worker, *a) for a in args]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                log(f"  bucket {r['bucket']:2d}: {r['pairs']:,} pairs, {len(r['features'])} features" + (f", {r['positives']:,} positives" if r["positives"] is not None else "") +
                    f"  [{r['seconds']:.0f}s, child peak {r['peak_gb']} GB, total {time.time() - t0:.0f}s, {len(results)}/{len(args)} done]")
    results.sort(key=lambda r: r["bucket"])
    cols = results[-1]["features"]
    meta = {"n_pairs": int(n_pairs), "n_buckets": n_buckets, "features": cols, "n_features": len(cols), "groups": results[-1]["groups"],
            "stats": stats_info, "buckets": [{k: v for k, v in r.items() if k not in ("features", "groups")} for r in results], "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out_dir, "features_meta.json"), "w"), indent=1)
    log(f"done: {len(cols)} features, {sum(r['pairs'] for r in results):,} pairs in {meta['seconds']:.0f}s -> {out_dir}")
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--pairs", required=True, help="parquet with q_idx/p_idx (positions in the Source 1 subset and in the S2+S3 concatenation)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--entity-ids", default=None)
    p.add_argument("--labels", default=None, help="parquet with the true pairs' q_idx/p_idx (adds a label column)")
    p.add_argument("--buckets", type=int, default=32)
    p.add_argument("--only-bucket", type=int, default=None)
    p.add_argument("--procs", type=int, default=-1, help="threads for the string metrics inside a bucket (-1 = all)")
    p.add_argument("--workers", type=int, default=1, help="bucket processes running concurrently (about 4 GB each)")
    p.add_argument("--model", default=None, help="LightGBM model file: score the pairs instead of storing the features (writes pred_XX.parquet)")
    p.add_argument("--feature-list", default=None, help="features_meta.json of the training run (the model's feature order); required with --model")
    p.add_argument("--neg-rate", type=float, default=1.0, help="negative subsampling rate the model was trained with (prior correction)")
    a = p.parse_args(argv)
    model = None
    if a.model:
        feats = json.load(open(a.feature_list))["features"]
        feats = [f for f in feats if f != "country_eq"]
        model = {"path": a.model, "features": feats, "neg_rate": a.neg_rate}
    ids = None
    if a.entity_ids:
        with open(a.entity_ids, encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    t0 = time.time()
    build(a.pre_dir, a.prefix, a.pairs, a.out_dir, ids, a.labels, a.buckets, a.procs, log=lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True),
          only_bucket=a.only_bucket, n_procs=a.workers, model=model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
