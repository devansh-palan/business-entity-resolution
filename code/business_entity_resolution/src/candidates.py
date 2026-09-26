"""Candidate generation (blocking) on the preprocessed tables.

Every strategy returns a pair table (q_idx, p_idx) of row positions: ``q_idx`` in the query (Source 1) frame and
``p_idx`` in the pool (Source 2 + 3) frame.  Both frames carry those positions as columns (see ``with_positions``),
join keys are hashed to 64-bit integers, and pairs are 8 bytes each, so a 20M-pair block costs ~200 MB instead of
the gigabytes that string-keyed joins need; ids are mapped back only when the competition file is written.
All keys include ``country`` as an opaque string (no country is hard-coded).  Strategies:

* key blocks            exact equality on one or more parsed columns, optionally dropping keys carried by too many
                        pool records
* token blocks          shared name / address token whose pool document frequency is under a cap, optionally only
                        the rarest tokens of each query
* sparse retrieval      TF-IDF character n-grams (names) or tokens (addresses, name + address) with top-k cosine
                        neighbours per query and per source, computed as sparse top-n matrix products
* sorted neighbourhood  pool records adjacent to the query in the sorted order of a key
* union, per-true-pair coverage, volume/recall measurement and the competition output file

The vectorisers are numpy-based so that 10M records vectorise in seconds; the top-k product uses sparse_dot_topn.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIR_COLS = ["q_idx", "p_idx"]
PAIR_SCHEMA = {"q_idx": pl.UInt32, "p_idx": pl.UInt32}
POOL_PARTS = 4   # the pool is processed in row slices so that no exploded or joined intermediate exceeds a quarter of it


def empty_pairs() -> pl.DataFrame:
    return pl.DataFrame(schema=PAIR_SCHEMA)


def with_positions(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Add the row position column (``q_idx`` for queries, ``p_idx`` for the pool) if it is missing."""
    return df if name in df.columns else df.with_row_index(name).with_columns(pl.col(name).cast(pl.UInt32))


def _hash_key(keys: Sequence[str]) -> pl.Expr:
    return pl.struct(["country", *keys]).hash(seed=0).alias("k")


def _parts(df: pl.DataFrame, n: int = POOL_PARTS) -> List[pl.DataFrame]:
    size = -(-df.height // n) if df.height else 1
    return [df.slice(i, size) for i in range(0, df.height, size)]


def pack(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(q_idx, p_idx) -> one uint64 key."""
    return (q.astype(np.uint64) << np.uint64(32)) | p.astype(np.uint64)


def unpack(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (keys >> np.uint64(32)).astype(np.uint32), (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def pairs_frame(q: np.ndarray, p: np.ndarray, dedupe: bool = True) -> pl.DataFrame:
    if len(q) == 0:
        return empty_pairs()
    if dedupe:
        q, p = unpack(np.unique(pack(q, p)))
    return pl.DataFrame({"q_idx": q.astype(np.uint32), "p_idx": p.astype(np.uint32)})


def _pairs(parts: List[pl.DataFrame]) -> pl.DataFrame:
    parts = [p.select(PAIR_COLS) for p in parts if p.height]
    if not parts:
        return empty_pairs()
    q = np.concatenate([p["q_idx"].to_numpy() for p in parts])
    p_ = np.concatenate([p["p_idx"].to_numpy() for p in parts])
    return pairs_frame(q, p_)


def np_join(q_keys: np.ndarray, q_idx: np.ndarray, p_keys: np.ndarray, p_idx: np.ndarray, max_block: Optional[int] = None,
            chunk_out: int = 50_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Equi-join of query keys against pool keys with sorted search: (q_idx, p_idx) arrays of every match.

    Pool keys are sorted once; each query expands to its [lo, hi) range.  ``max_block`` drops keys carried by more
    than that many pool records.  Memory is the sorted pool (12 bytes per record) plus the output.
    """
    if len(q_keys) == 0 or len(p_keys) == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    order = np.argsort(p_keys, kind="stable")
    ps, pi = p_keys[order], p_idx[order]
    del order
    lo = np.searchsorted(ps, q_keys, "left")
    hi = np.searchsorted(ps, q_keys, "right")
    cnt = (hi - lo).astype(np.int64)
    if max_block is not None:
        cnt[cnt > max_block] = 0
    keep = cnt > 0
    lo, cnt, qi = lo[keep], cnt[keep], q_idx[keep]
    if len(cnt) == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    outs_q, outs_p = [], []
    start = 0
    csum = np.cumsum(cnt)
    while start < len(cnt):                                  # bounded output chunks
        base = int(csum[start - 1]) if start else 0
        end = int(np.searchsorted(csum, base + chunk_out, "right"))
        end = max(end, start + 1)
        c = cnt[start:end]
        tot = int(c.sum())
        starts = np.repeat(lo[start:end], c)
        offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
        outs_q.append(np.repeat(qi[start:end], c).astype(np.uint32))
        outs_p.append(pi[starts + offs])
        start = end
    return np.concatenate(outs_q), np.concatenate(outs_p).astype(np.uint32)


def key_block(s1: pl.DataFrame, pool: pl.DataFrame, keys: Sequence[str], max_block: Optional[int] = None) -> pl.DataFrame:
    """Pairs sharing country and every key column (nulls never match).  ``max_block`` drops keys carried by more
    pool records than that (the block would be uninformative and expensive)."""
    keys = list(keys)
    a = s1.drop_nulls(keys).select("q_idx", _hash_key(keys))
    b = pool.drop_nulls(keys).select("p_idx", _hash_key(keys))
    q, p = np_join(a["k"].to_numpy(), a["q_idx"].to_numpy(), b["k"].to_numpy(), b["p_idx"].to_numpy(), max_block)
    return pairs_frame(q, p, dedupe=False)          # one pool row per (query, key): already unique


def multi_key_block(s1: pl.DataFrame, pool: pl.DataFrame, key_sets: Sequence[Sequence[str]], max_block: Optional[int] = None) -> pl.DataFrame:
    return union([key_block(s1, pool, ks, max_block) for ks in key_sets])


def _explode_np(part: pl.DataFrame, idx: str, col: str, min_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """(idx, k) arrays, one entry per distinct token of each record, k = hash of (country, token)."""
    ex = (part.select(idx, "country", pl.col(col).alias("tok")).explode("tok").drop_nulls("tok")
              .filter(pl.col("tok").str.len_chars() >= min_len).select(idx, pl.struct(["country", "tok"]).hash(seed=0).alias("k")).unique())
    return ex[idx].to_numpy().astype(np.uint32), ex["k"].to_numpy()


def _member(keys: np.ndarray, sorted_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Membership of keys in a sorted reference array: (mask, position in the reference)."""
    if len(sorted_ref) == 0:
        return np.zeros(len(keys), bool), np.zeros(len(keys), np.int64)
    pos = np.searchsorted(sorted_ref, keys)
    pos[pos >= len(sorted_ref)] = 0
    return sorted_ref[pos] == keys, pos


class TokenIndex:
    """The pool exploded once into (record, token-hash) arrays per slice, with the document frequency of every token
    (within its country).  Several token blocks with different caps share one index."""

    def __init__(self, pool: Optional[pl.DataFrame], col: str, min_len: int = 1, n_parts: int = 8, parts: Optional[Iterable[pl.DataFrame]] = None):
        """``pool`` is sliced in memory, or ``parts`` (frames with p_idx, country, col) are consumed one at a time."""
        self.col, self.min_len = col, min_len
        self.parts: List[Tuple[np.ndarray, np.ndarray]] = []
        ks, cs = [], []
        source = parts if parts is not None else _parts(pool.select("p_idx", "country", col), n_parts)
        for part in source:
            p_idx, k = _explode_np(part.select("p_idx", "country", col), "p_idx", col, min_len)
            self.parts.append((p_idx, k))
            u, c = np.unique(k, return_counts=True)
            ks.append(u)
            cs.append(c)
        if ks:
            allk, allc = np.concatenate(ks), np.concatenate(cs)
            self.keys, inv = np.unique(allk, return_inverse=True)
            self.df = np.bincount(inv, weights=allc).astype(np.int64)
            del allk, allc, inv
        else:
            self.keys, self.df = np.zeros(0, np.uint64), np.zeros(0, np.int64)

    def block(self, s1: pl.DataFrame, df_cap: int, rarest: Optional[int] = None, max_pairs: int = 120_000_000) -> pl.DataFrame:
        """Pairs sharing a token with pool df <= df_cap; ``rarest`` keeps the n rarest qualifying tokens per query."""
        mask = self.df <= df_cap
        rare_keys, rare_df = self.keys[mask], self.df[mask]
        q_idx, q_k = _explode_np(s1.select("q_idx", "country", self.col), "q_idx", self.col, self.min_len)
        m, pos = _member(q_k, rare_keys)
        q_idx, q_k, q_df = q_idx[m], q_k[m], rare_df[pos[m]]
        if rarest is not None and len(q_idx):
            order = np.lexsort((q_k, q_df, q_idx))            # by query, then df ascending
            q_idx, q_k = q_idx[order], q_k[order]
            first = np.r_[0, np.flatnonzero(np.diff(q_idx.astype(np.int64))) + 1]
            sizes = np.diff(np.r_[first, len(q_idx)])
            rank = np.arange(len(q_idx)) - np.repeat(first, sizes)
            q_idx, q_k = q_idx[rank < rarest], q_k[rank < rarest]
        keys = []
        total = 0
        for p_idx, p_k in self.parts:
            m, _ = _member(p_k, rare_keys)
            q, p = np_join(q_k, q_idx, p_k[m], p_idx[m], chunk_out=10_000_000)
            keys.append(np.unique(pack(q, p)))           # de-duplicated per slice; slices are disjoint pool rows
            total += len(keys[-1])
            del q, p
            if total > max_pairs:
                raise MemoryError(f"token block exceeds {max_pairs:,} pairs (df_cap={df_cap}, rarest={rarest}); lower the cap")
        if not keys:
            return empty_pairs()
        q, p = unpack(np.concatenate(keys))
        return pairs_frame(q, p, dedupe=False)


def token_block(s1: pl.DataFrame, pool: pl.DataFrame, col: str, df_cap: int, rarest: Optional[int] = None,
                min_len: int = 1, max_pairs: int = 120_000_000) -> pl.DataFrame:
    """Pairs sharing a token of list column ``col`` whose pool document frequency (within the country) is <= df_cap.
    See TokenIndex; build the index yourself to evaluate several caps on one explode."""
    return TokenIndex(pool, col, min_len).block(s1, df_cap, rarest, max_pairs)


def sorted_neighbourhood(s1: pl.DataFrame, pool: pl.DataFrame, key: str, window: int = 3) -> pl.DataFrame:
    """For every query, the ``window`` pool records before and after it in the (country, key) sorted order."""
    a = s1.drop_nulls([key]).select(pl.col("q_idx").alias("id"), "country", key).with_columns(pl.lit(True).alias("is_q"))
    b = pool.drop_nulls([key]).select(pl.col("p_idx").alias("id"), "country", key).with_columns(pl.lit(False).alias("is_q"))
    allr = pl.concat([a, b]).sort(["country", key, "is_q"]).with_columns((~pl.col("is_q")).cast(pl.Int64).cum_sum().alias("rank"))
    # rank = number of pool records up to this row; a query at rank r neighbours pool ranks r-window+1 .. r+window
    pool_pos = allr.filter(~pl.col("is_q")).select("country", pl.col("id").alias("p_idx"), "rank")
    queries = allr.filter(pl.col("is_q")).select("country", pl.col("id").alias("q_idx"), "rank")
    offsets = pl.DataFrame({"off": list(range(-window + 1, window + 1))})
    q = queries.join(offsets, how="cross").with_columns((pl.col("rank") + pl.col("off")).alias("rank")).drop("off")
    return _pairs([q.join(pool_pos, on=["country", "rank"], how="inner")])


# --------------------------------------------------------------------------- #
# Sparse vectorisers
# --------------------------------------------------------------------------- #
class CharNgramVectorizer:
    """Character 3-gram TF-IDF over UTF-8 bytes, padded with a space on both sides, built with numpy.

    fit() learns document frequencies on the pool; columns with df < min_df or df > max_df * N are dropped.
    Rows are L2-normalised sublinear TF-IDF, so a sparse product gives cosine similarity.
    """

    def __init__(self, n: int = 3, width: int = 48, min_df: int = 2, max_df: float = 0.01):
        assert n == 3, "only 3-grams are implemented (codes fit in 24 bits)"
        self.n, self.width, self.min_df, self.max_df = n, width, min_df, max_df
        self.col_of: Optional[np.ndarray] = None      # code -> column id or -1
        self.idf: Optional[np.ndarray] = None
        self.n_features = 0

    def _codes(self, texts: Sequence[Optional[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """(row index, 3-gram code) for every 3-gram of every text, one entry per occurrence."""
        w = self.width
        buf = np.zeros((len(texts), w), dtype=np.uint8)
        for i, t in enumerate(texts):
            if not t:
                continue
            b = (" " + t + " ").encode("utf-8")[:w]
            buf[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        a = buf.astype(np.int32)
        codes = (a[:, :-2] << 16) | (a[:, 1:-1] << 8) | a[:, 2:]
        valid = a[:, 2:] != 0
        rows = np.repeat(np.arange(len(texts), dtype=np.int64), w - 2).reshape(len(texts), w - 2)[valid]
        return rows, codes[valid]

    @staticmethod
    def _unique_pairs(rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = rows.astype(np.int64) * (1 << 24) + cols
        uk, tf = np.unique(key, return_counts=True)
        return uk >> 24, (uk & ((1 << 24) - 1)).astype(np.int64), tf

    def fit(self, texts: Sequence[Optional[str]], chunk: int = 500_000) -> "CharNgramVectorizer":
        df = np.zeros(1 << 24, dtype=np.int64)
        for i in range(0, len(texts), chunk):
            r, c = self._codes(texts[i:i + chunk])
            _, cols, _ = self._unique_pairs(r, c)
            df += np.bincount(cols, minlength=1 << 24)
        n = max(len(texts), 1)
        keep = (df >= self.min_df) & (df <= self.max_df * n)
        self.col_of = np.full(1 << 24, -1, dtype=np.int64)
        self.col_of[keep] = np.arange(int(keep.sum()))
        self.n_features = int(keep.sum())
        self.idf = np.log((1 + n) / (1 + df[keep])) + 1.0
        return self

    def transform(self, texts: Sequence[Optional[str]], chunk: int = 500_000) -> sp.csr_matrix:
        parts = []
        for i in range(0, len(texts), chunk):
            r, c = self._codes(texts[i:i + chunk])
            rows, cols, tf = self._unique_pairs(r, c)
            col = self.col_of[cols]
            m = col >= 0
            data = (1.0 + np.log(tf[m])) * self.idf[col[m]]
            mat = sp.csr_matrix((data.astype(np.float32), (rows[m], col[m])), shape=(len(texts[i:i + chunk]), self.n_features))
            parts.append(mat)
        out = sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, self.n_features), dtype=np.float32)
        return _l2_normalise(out)


class TokenVectorizer:
    """TF-IDF over pre-tokenised list columns (addresses, name + address), vocabulary learned on the pool."""

    def __init__(self, min_df: int = 2, max_df: float = 0.01):
        self.min_df, self.max_df = min_df, max_df
        self.vocab: Optional[pl.DataFrame] = None
        self.idf: Optional[np.ndarray] = None
        self.n_features = 0

    @staticmethod
    def _explode(tokens: pl.Series) -> pl.DataFrame:
        return (pl.DataFrame({"row": np.arange(tokens.len(), dtype=np.int64), "tok": tokens}).explode("tok").drop_nulls("tok")
                  .group_by(["row", "tok"]).agg(pl.len().alias("tf")))

    def fit(self, tokens: pl.Series, chunk: int = 500_000) -> "TokenVectorizer":
        """Document frequencies from the unique tokens of every record, accumulated in chunks (no (row, token) hash table)."""
        n = max(tokens.len(), 1)
        parts = []
        for i in range(0, tokens.len(), chunk):
            part = tokens.slice(i, chunk)
            parts.append(pl.DataFrame({"tok": part.list.unique()}).explode("tok").drop_nulls("tok").group_by("tok").agg(pl.len().alias("df")))
        df = (pl.concat(parts).group_by("tok").agg(pl.col("df").sum()) if parts else pl.DataFrame({"tok": [], "df": []}, schema={"tok": pl.String, "df": pl.UInt32}))
        df = df.filter((pl.col("df") >= self.min_df) & (pl.col("df") <= self.max_df * n)).sort("tok")
        self.vocab = df.with_columns(pl.arange(0, df.height, dtype=pl.Int64).alias("col"))
        self.n_features = self.vocab.height
        self.idf = (np.log((1 + n) / (1 + self.vocab["df"].to_numpy())) + 1.0).astype(np.float32)
        return self

    def transform(self, tokens: pl.Series, chunk: int = 500_000) -> sp.csr_matrix:
        """Sublinear TF-IDF rows, L2-normalised, built one chunk of records at a time."""
        parts = []
        for i in range(0, tokens.len(), chunk):
            part = tokens.slice(i, chunk)
            ex = self._explode(part).join(self.vocab.select("tok", "col"), on="tok", how="inner")
            rows, cols, tf = ex["row"].to_numpy(), ex["col"].to_numpy(), ex["tf"].to_numpy().astype(np.float32)
            data = (1.0 + np.log(tf)) * self.idf[cols]
            parts.append(sp.csr_matrix((data, (rows, cols)), shape=(part.len(), self.n_features), dtype=np.float32))
            del ex
        out = sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, self.n_features), dtype=np.float32)
        return _l2_normalise(out)


def _l2_normalise(m: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms, dtype=np.float32) @ m


# --------------------------------------------------------------------------- #
# Top-k sparse retrieval
# --------------------------------------------------------------------------- #
def topk_pairs(q_idx: np.ndarray, Q: sp.csr_matrix, p_idx: np.ndarray, P: sp.csr_matrix, k: int = 20, threshold: float = 0.2,
               n_threads: int = 8, chunk: int = 50_000) -> pl.DataFrame:
    """Top-k cosine neighbours of every query row among the pool rows: pairs with ``score`` and ``rank`` (1 = best)."""
    from sparse_dot_topn import sp_matmul_topn
    PT = P.T.tocsr()
    q_idx = np.asarray(q_idx, dtype=np.uint32)
    p_idx = np.asarray(p_idx, dtype=np.uint32)
    parts = []
    for i in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[i:i + chunk], PT, top_n=k, threshold=threshold, sort=True, n_threads=n_threads).tocsr()
        counts = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0]), counts)
        rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts) + 1
        parts.append(pl.DataFrame({"q_idx": q_idx[i + rows], "p_idx": p_idx[C.indices], "score": C.data.astype(np.float32), "rank": rank.astype(np.int32)}))
    if not parts:
        return pl.DataFrame(schema={**PAIR_SCHEMA, "score": pl.Float32, "rank": pl.Int32})
    return pl.concat(parts)


def retrieve_per_country_source(s1: pl.DataFrame, pool: Optional[pl.DataFrame], make_vectorizer: Callable, text_of: Callable, k: int = 20,
                                threshold: float = 0.2, n_threads: int = 8, log=None, slice_loader: Optional[Callable] = None,
                                sources: Sequence[str] = ("S2", "S3"), spill_dir: Optional[str] = None) -> pl.DataFrame:
    """Run a top-k retrieval separately for every (country, source) pool slice, so each query gets up to k candidates
    from Source 2 and up to k from Source 3 of its own country.  ``text_of(df)`` returns the texts/tokens to vectorise.
    The slice comes from ``pool`` (filtered in memory) or from ``slice_loader(country, src)`` (streamed), which keeps
    only one slice's text in memory at a time; with ``spill_dir`` each slice's result is written to disk as soon as it
    is computed and the frames are re-read only for the final concatenation."""
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        qs = s1.filter(pl.col("country") == country)
        if qs.height == 0:
            continue
        for src in sources:
            ps = slice_loader(country, src) if slice_loader is not None else pool.filter((pl.col("country") == country) & (pl.col("src") == src))
            if ps.height == 0:
                continue
            vec = make_vectorizer().fit(text_of(ps))
            P = vec.transform(text_of(ps))
            p_idx = ps["p_idx"].to_numpy()
            del ps
            Q = vec.transform(text_of(qs))
            res = topk_pairs(qs["q_idx"].to_numpy(), Q, p_idx, P, k=k, threshold=threshold, n_threads=n_threads)
            if log:
                log(f"    retrieval {country}/{src}: {qs.height:,} queries x {len(p_idx):,} pool, {vec.n_features:,} features -> {res.height:,} pairs")
            if spill_dir is not None:                  # keep only one slice's result in memory
                os.makedirs(spill_dir, exist_ok=True)
                path = os.path.join(spill_dir, f"{country}_{src}.parquet")
                res.write_parquet(path)
                out.append(path)
            else:
                out.append(res)
            del P, Q, vec, res
    if not out:
        return pl.DataFrame(schema={**PAIR_SCHEMA, "score": pl.Float32, "rank": pl.Int32})
    return pl.concat([pl.read_parquet(p) for p in out]) if spill_dir is not None else pl.concat(out)


# --------------------------------------------------------------------------- #
# Union, coverage, measurement, output
# --------------------------------------------------------------------------- #
def union(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    return _pairs([f for f in frames if f is not None])


def _keys_of(pairs: pl.DataFrame) -> np.ndarray:
    return pack(pairs["q_idx"].to_numpy(), pairs["p_idx"].to_numpy())


def coverage(positives: pl.DataFrame, strategies: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per true pair (q_idx, p_idx[, src]) with a boolean column per strategy."""
    out = positives.select([*PAIR_COLS, *[c for c in ("src",) if c in positives.columns]])
    pos_keys = _keys_of(out)
    for name, pairs in strategies.items():
        out = out.with_columns(pl.Series(name, np.isin(pos_keys, _keys_of(pairs), kind="sort")))
    return out


def measure(pairs: pl.DataFrame, positives: pl.DataFrame, n_s1: int, n_pool: int, country_of: pl.DataFrame,
            src_boundary: Optional[int] = None, assume_unique: bool = False) -> dict:
    """Recall and volume of a candidate set.

    positives    : true pairs (q_idx, p_idx, src) of the query entities
    country_of   : (q_idx, country) of the query entities
    src_boundary : pool rows below it are Source 2, the rest Source 3 (for the pairs-by-source count)
    Volume statistics count every query entity, including those with no candidate.  ``assume_unique`` skips the
    de-duplication when the caller guarantees unique (q_idx, p_idx) rows.
    """
    keys = _keys_of(pairs)
    if not assume_unique:
        keys = np.unique(keys)
    q_all, p_all = unpack(keys)
    hit = positives.with_columns(pl.Series("hit", np.isin(_keys_of(positives), keys, kind="sort"))).join(country_of, on="q_idx", how="left")
    per_entity = hit.group_by("q_idx").agg(pl.col("hit").all().alias("all_found"))
    vol = np.bincount(q_all, minlength=n_s1) if len(q_all) else np.zeros(n_s1, dtype=np.int64)
    by = lambda col: {str(r[col]): round(float(r["r"]), 5) for r in hit.group_by(col).agg(pl.col("hit").mean().alias("r")).sort(col).to_dicts()}
    per_src_pairs = {}
    if src_boundary is not None:
        n_s2 = int((p_all < src_boundary).sum())
        per_src_pairs = {"S2": n_s2, "S3": int(len(keys) - n_s2)}
    return {
        "pairs": int(len(keys)),
        "recall": round(float(hit["hit"].mean()), 5),
        "true_pairs_found": int(hit["hit"].sum()),
        "true_pairs_missed": int((~hit["hit"]).sum()),
        "entity_all_matches_found": round(float(per_entity["all_found"].mean()), 5),
        "recall_by_source": by("src") if "src" in hit.columns else {},
        "recall_by_country": by("country"),
        "mean": round(float(vol.mean()), 3), "median": float(np.median(vol)), "p95": float(np.percentile(vol, 95)), "max": int(vol.max()),
        "entities_with_candidates": int((vol > 0).sum()),
        "reduction_ratio": round(1.0 - len(keys) / (n_s1 * n_pool), 6),
        "pairs_by_source": per_src_pairs,
    }


def cap_per_entity(pairs: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    """Keep the k best-scoring candidates of every entity (ties broken by p_idx for determinism)."""
    return pairs.sort(["q_idx", score_col, "p_idx"], descending=[False, True, False]).group_by("q_idx", maintain_order=True).head(k)


def pairs_to_ids(pairs: pl.DataFrame, s1_ids: Sequence[str], pool_ids: Sequence[str]) -> pl.DataFrame:
    """Map (q_idx, p_idx) rows to (source1_entity_id, matched_id)."""
    q = pl.DataFrame({"q_idx": np.arange(len(s1_ids), dtype=np.uint32), "source1_entity_id": list(s1_ids)})
    p = pl.DataFrame({"p_idx": np.arange(len(pool_ids), dtype=np.uint32), "matched_id": list(pool_ids)})
    return pairs.select(PAIR_COLS).join(q, on="q_idx", how="inner").join(p, on="p_idx", how="inner").select("source1_entity_id", "matched_id")


def write_candidate_pairs(pairs: pl.DataFrame, s1_ids: Sequence[str], pool_ids: Sequence[str], path: str, id_column: str = "candidate_entity_ids") -> int:
    """Write candidate_pairs.tsv per the specification: header, tab-separated, LF, no quoting, one row per Source 1
    entity in the given order, ids sorted and comma-joined, empty when there is none.  Streams row by row from
    numpy arrays, so a file of 100M ids needs no table of 100M strings.  Returns the number of rows written.
    ``id_column='matched_entity_ids'`` writes matching_results.tsv in the same layout."""
    keys = np.unique(_keys_of(pairs)) if pairs.height else np.zeros(0, np.uint64)
    q, p = unpack(keys)
    bounds = np.searchsorted(q, np.arange(len(s1_ids) + 1, dtype=np.uint32))
    pool_arr = np.asarray(list(pool_ids), dtype=object)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_column}\n")
        for i, sid in enumerate(s1_ids):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            ids = ",".join(sorted(pool_arr[p[lo:hi]].tolist())) if hi > lo else ""
            f.write(f"{sid}\t{ids}\n")
    return len(s1_ids)


def validate_candidate_file(path: str, s1_ids: Sequence[str], pool_id_set: set) -> dict:
    """Streaming check of a candidate_pairs.tsv against the specification: header, one row per Source 1 entity in
    order, S2-/S3- ids only, no duplicates within a row, every id in the pool, LF endings.  Returns counts and the
    first problems found (empty list = valid)."""
    problems: List[str] = []
    n_rows = n_ids = n_empty = 0
    with open(path, "rb") as f:
        header = f.readline()
        if header != b"source1_entity_id\tcandidate_entity_ids\n":
            problems.append(f"header is {header!r}")
        for i, raw in enumerate(f):
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                problems.append(f"row {i + 1}: line ending is not LF")
            line = raw.decode("utf-8").rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 2:
                problems.append(f"row {i + 1}: {len(parts)} columns")
                continue
            sid, ids = parts
            if i >= len(s1_ids) or sid != s1_ids[i]:
                problems.append(f"row {i + 1}: expected {s1_ids[i] if i < len(s1_ids) else 'no row'}, found {sid}")
            n_rows += 1
            if ids == "":
                n_empty += 1
                continue
            lst = ids.split(",")
            n_ids += len(lst)
            if len(set(lst)) != len(lst):
                problems.append(f"row {i + 1}: duplicate ids")
            for m in lst:
                if not (m.startswith("S2-") or m.startswith("S3-")):
                    problems.append(f"row {i + 1}: id {m} is not S2-/S3-")
                    break
                if m not in pool_id_set:
                    problems.append(f"row {i + 1}: id {m} not in the pool")
                    break
            if len(problems) > 20:
                break
    if n_rows != len(s1_ids):
        problems.append(f"{n_rows} rows for {len(s1_ids)} entities")
    return {"rows": n_rows, "ids": n_ids, "empty_rows": n_empty, "problems": problems[:20], "valid": not problems}
