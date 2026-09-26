"""Generate candidate_pairs.tsv for a preprocessed split with the strategy set selected on validation fold 0.

The final set (eda/candidate_generation.md, extended after the miss analysis of eda/candidate_misses.md) is the
union of thirteen strategies, all keyed by the country string:
  * name + address token TF-IDF retrieval, top 20 per (country, source)
  * address token TF-IDF retrieval, ranks 1-5 of the top-20 run per (country, source)
  * name 3-gram TF-IDF retrieval against the pool records that have no address, top 5 per (country, source)
  * exact name in any of the four representations, blocks over 200 pool records dropped
  * exact name (any representation) + house number
  * exact name (any representation) + city, blocks over 100 pool records dropped
  * compact name + state
  * state + house number + street
  * house number + street + city, blocks over 200 pool records dropped
  * state + sorted name tokens
  * city + first name token, blocks over 100 pool records dropped
  * sorted neighbourhood (window 3) on the compact name
  * sorted neighbourhood (window 3) on the sorted address tokens
Nothing here uses labels; the same code produced the fold-0 file that was measured against the truth.

Usage (from the project root):
    python code/business_entity_resolution/src/generate_candidates.py --pre-dir artifacts/preprocessed/all --prefix test \
        --out output/candidate_pairs.tsv [--entity-ids ids.txt] [--procs 8]
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Sequence

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates as C  # noqa: E402

NAME_REPS = ("name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted")
DERIVED = {   # derived column -> (base columns, expression)
    "name_first": (["name_core_tokens"], pl.col("name_core_tokens").list.first()),
    "name_addr_tokens": (["name_translit_core", "addr_tokens"], pl.concat_list([pl.col("name_translit_core").str.split(" "), pl.col("addr_tokens")])),
}
EXACT_NAME_STRATEGIES = ("exact_name_any_cap200", "name_any_house_number", "name_any_city_cap100")
KEYED_STRATEGIES = ("state_house_street", "state_name_sorted", "city_name_first_cap100", "sn_name_compact_w3", "sn_addr_sorted_w3",
                    "compact_name_state", "house_street_city_cap200")
FINAL_STRATEGIES = ["name_addr_retrieval_top20", "addr_token_retrieval_top5", "addrless_name_ngram_top5", "exact_name_any_cap200",
                    "name_any_house_number", "name_any_city_cap100", "compact_name_state", "state_house_street", "house_street_city_cap200",
                    "state_name_sorted", "city_name_first_cap100", "sn_name_compact_w3", "sn_addr_sorted_w3"]
FINAL_STRATEGIES_V1 = ["name_addr_retrieval_top20", "exact_name_any_cap200", "state_house_street", "state_name_sorted", "city_name_first_cap100",
                       "sn_name_compact_w3", "sn_addr_sorted_w3"]     # the seven-strategy set of candidate_generation.md (recall 0.9852 on fold 0)


def derive(df: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    return df.with_columns(*[DERIVED[n][1].alias(n) for n in names if n in DERIVED])


def _base(cols: Sequence[str]) -> List[str]:
    return list(dict.fromkeys([b for c in cols for b in (DERIVED[c][0] if c in DERIVED else [c])]))


def pool_files(pre_dir: str, prefix: str) -> List[str]:
    return [os.path.join(pre_dir, f"{prefix}_source{k}.parquet") for k in (2, 3)]


def load_queries(pre_dir: str, prefix: str, cols: Sequence[str], entity_ids: Optional[Sequence[str]] = None) -> pl.DataFrame:
    df = pl.read_parquet(os.path.join(pre_dir, f"{prefix}_source1.parquet"), columns=["entity_id", "country", *_base(cols)])
    if entity_ids is not None:
        df = df.join(pl.DataFrame({"entity_id": list(entity_ids)}), on="entity_id", how="inner")
    return C.with_positions(derive(df, cols), "q_idx")


def load_pool(files: Sequence[str], cols: Sequence[str]) -> pl.DataFrame:
    df = pl.concat([pl.read_parquet(f, columns=["entity_id", "src", "country", *_base(cols)]) for f in files])
    return C.with_positions(derive(df, cols), "p_idx").select(["p_idx", "entity_id", "src", "country", *cols])


def load_pool_slice(files: Sequence[str], cols: Sequence[str], country: str, src: str) -> pl.DataFrame:
    """One (country, source) slice with global p_idx positions, read straight from parquet."""
    offset = 0
    for k, f in zip((2, 3), files):
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        if f"S{k}" == src:
            part = (pl.scan_parquet(f).with_row_index("p_idx", offset=offset).filter(pl.col("country") == country)
                      .select(["p_idx", "country", *_base(cols)]).collect().with_columns(pl.col("p_idx").cast(pl.UInt32)))
            return derive(part, cols).select(["p_idx", "country", *cols])
        offset += n
    return pl.DataFrame(schema={"p_idx": pl.UInt32})


def pool_ids(files: Sequence[str]) -> List[str]:
    return pl.concat([pl.read_parquet(f, columns=["entity_id"]) for f in files])["entity_id"].to_list()


RETRIEVAL_KINDS = {   # what each TF-IDF retrieval vectorises, and which pool records it searches
    "name_addr": {"cols": ["name_addr_tokens"], "vectorizer": lambda: C.TokenVectorizer(max_df=0.01), "text": lambda d: d["name_addr_tokens"], "filter": None},
    "addr": {"cols": ["addr_tokens"], "vectorizer": lambda: C.TokenVectorizer(max_df=0.01), "text": lambda d: d["addr_tokens"], "filter": None},
    "addrless_ngram": {"cols": ["name_translit_core", "addr_missing"], "vectorizer": lambda: C.CharNgramVectorizer(max_df=0.05),
                       "text": lambda d: d["name_translit_core"].to_list(), "filter": "addr_missing"},
}
_EMPTY_RETRIEVAL = {**C.PAIR_SCHEMA, "score": pl.Float32, "rank": pl.Int32}


def _retrieval_slice_worker(pre_dir: str, prefix: str, entity_ids: Optional[Sequence[str]], kind: str, country: str, src: str, k: int, threshold: float,
                            n_threads: int, out_path: str) -> dict:
    """One (country, source) slice of a retrieval in its own process: fit on the pool slice, score the country's queries,
    write (q_idx, p_idx, score, rank) to ``out_path``.  The process ends afterwards, so its memory goes back to the OS."""
    t0 = time.time()
    spec = RETRIEVAL_KINDS[kind]
    files = pool_files(pre_dir, prefix)
    qs = load_queries(pre_dir, prefix, spec["cols"], entity_ids).filter(pl.col("country") == country)
    ps = load_pool_slice(files, spec["cols"], country, src)
    if spec["filter"] and ps.height:
        ps = ps.filter(pl.col(spec["filter"]))
    info = {"country": country, "src": src, "queries": int(qs.height), "pool": int(ps.height), "features": 0, "pairs": 0}
    if qs.height == 0 or ps.height == 0:
        pl.DataFrame(schema=_EMPTY_RETRIEVAL).write_parquet(out_path)
        return info
    vec = spec["vectorizer"]().fit(spec["text"](ps))
    info["features"] = int(vec.n_features)
    if vec.n_features == 0:
        pl.DataFrame(schema=_EMPTY_RETRIEVAL).write_parquet(out_path)
        return info
    P = vec.transform(spec["text"](ps))
    p_idx = ps["p_idx"].to_numpy()
    del ps
    Q = vec.transform(spec["text"](qs))
    res = C.topk_pairs(qs["q_idx"].to_numpy(), Q, p_idx, P, k=k, threshold=threshold, n_threads=n_threads)
    res.write_parquet(out_path)
    info.update({"pairs": int(res.height), "seconds": round(time.time() - t0, 1)})
    try:
        import psutil
        info["peak_gb"] = round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return info


def retrieve_in_processes(pre_dir: str, prefix: str, entity_ids: Optional[Sequence[str]], kind: str, k: int, threshold: float, n_threads: int,
                          spill_dir: str, log: Callable = print) -> pl.DataFrame:
    """Top-k retrieval per (country, source), every slice in a fresh child process; results are spilled to parquet and
    concatenated at the end (same values as candidates.retrieve_per_country_source)."""
    os.makedirs(spill_dir, exist_ok=True)
    countries = sorted(pl.read_parquet(os.path.join(pre_dir, f"{prefix}_source1.parquet"), columns=["country"])["country"].unique().to_list())
    paths = []
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
        for country in countries:
            for src in ("S2", "S3"):
                out = os.path.join(spill_dir, f"{country}_{src}.parquet".replace("/", "_"))
                info = ex.submit(_retrieval_slice_worker, pre_dir, prefix, list(entity_ids) if entity_ids is not None else None, kind, country, src, k, threshold,
                                 n_threads, out).result()
                paths.append(out)
                log(f"    retrieval {kind} {country}/{src}: {info['queries']:,} queries x {info['pool']:,} pool, {info['features']:,} features -> {info['pairs']:,} pairs"
                    + (f" [{info['seconds']:.0f}s, peak {info.get('peak_gb', '?')} GB]" if "seconds" in info else ""))
    return pl.concat([pl.read_parquet(x) for x in paths])


def _block_worker(pre_dir: str, prefix: str, entity_ids: Optional[Sequence[str]], group: str, strategies: Sequence[str], out_dir: str) -> dict:
    """The exact-name blocks of one name representation (group 'exact:<rep>') or the keyed blocks and sorted neighbourhoods
    (group 'keyed'), computed in a child process; every block is written as (q_idx, p_idx) parquet under ``out_dir``."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    files = pool_files(pre_dir, prefix)
    out = {}

    def save(name: str, pairs: pl.DataFrame):
        pairs.select(C.PAIR_COLS).write_parquet(os.path.join(out_dir, f"{name}.parquet"))
        out[name] = int(pairs.height)

    if group.startswith("exact:"):
        rep = group.split(":", 1)[1]
        s1 = load_queries(pre_dir, prefix, [rep, "house_number", "city_norm"], entity_ids)
        pool = load_pool(files, [rep, "house_number", "city_norm"])
        if "exact_name_any_cap200" in strategies:
            save(f"exact_{rep}_cap200", C.key_block(s1, pool, [rep], max_block=200))
        if "name_any_house_number" in strategies:
            save(f"{rep}_house_number", C.key_block(s1, pool, [rep, "house_number"]))
        if "name_any_city_cap100" in strategies:
            save(f"{rep}_city_cap100", C.key_block(s1, pool, [rep, "city_norm"], max_block=100))
    elif group == "keyed":
        cols = ["house_number", "street", "state", "name_core_sorted", "city_norm", "name_first", "name_compact", "addr_sorted"]
        s1 = load_queries(pre_dir, prefix, cols, entity_ids)
        pool = load_pool(files, cols)
        if "state_house_street" in strategies:
            save("state_house_street", C.key_block(s1, pool, ["state", "house_number", "street"]))
        if "state_name_sorted" in strategies:
            save("state_name_sorted", C.key_block(s1, pool, ["state", "name_core_sorted"]))
        if "city_name_first_cap100" in strategies:
            save("city_name_first_cap100", C.key_block(s1, pool, ["city_norm", "name_first"], max_block=100))
        if "compact_name_state" in strategies:
            save("compact_name_state", C.key_block(s1, pool, ["name_compact", "state"]))
        if "house_street_city_cap200" in strategies:
            save("house_street_city_cap200", C.key_block(s1, pool, ["house_number", "street", "city_norm"], max_block=200))
        if "sn_name_compact_w3" in strategies:
            save("sn_name_compact_w3", C.sorted_neighbourhood(s1, pool, "name_compact", window=3))
        if "sn_addr_sorted_w3" in strategies:
            save("sn_addr_sorted_w3", C.sorted_neighbourhood(s1, pool, "addr_sorted", window=3))
    else:
        raise ValueError(group)
    info = {"blocks": out, "seconds": round(time.time() - t0, 1)}
    try:
        import psutil
        info["peak_gb"] = round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return info


def generate(pre_dir: str, prefix: str, out_path: str, entity_ids: Optional[Sequence[str]] = None, strategies: Sequence[str] = FINAL_STRATEGIES,
             k: int = 20, n_threads: int = 8, log: Callable = print) -> dict:
    """Run the selected strategies on a split, union them and write the competition file.  Returns run statistics."""
    t0 = time.time()
    files = pool_files(pre_dir, prefix)
    s1 = load_queries(pre_dir, prefix, [], entity_ids)
    s1_ids = s1["entity_id"].to_list()
    n_s1 = s1.height
    del s1
    log(f"queries {n_s1:,} from {prefix}_source1; pool files {files}")
    keys_acc: Optional[np.ndarray] = None            # the union so far, as packed (q_idx, p_idx) keys
    stats = {}

    def add(name: str, pairs: pl.DataFrame):
        nonlocal keys_acc
        stats[name] = {"pairs": int(pairs.height), "seconds": round(time.time() - t0, 1)}
        log(f"  {name:28s} pairs={pairs.height:>12,}  [{time.time() - t0:.0f}s]")
        new = np.unique(C._keys_of(pairs)) if pairs.height else np.zeros(0, dtype=np.uint64)
        keys_acc = new if keys_acc is None else np.union1d(keys_acc, new)
        del new
        gc.collect()

    spill = os.path.join(os.path.dirname(out_path) or ".", "_retrieval_spill")
    groups = [f"exact:{rep}" for rep in NAME_REPS if any(x in strategies for x in EXACT_NAME_STRATEGIES)] + (["keyed"] if any(x in strategies for x in KEYED_STRATEGIES) else [])
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:      # one fresh process per group: the parent stays thin
        for group in groups:
            gdir = os.path.join(spill, group.replace(":", "_"))
            info = ex.submit(_block_worker, pre_dir, prefix, list(entity_ids) if entity_ids is not None else None, group, list(strategies), gdir).result()
            for name in info["blocks"]:
                path = os.path.join(gdir, f"{name}.parquet")
                add(name, pl.read_parquet(path))
                os.remove(path)
            log(f"    group {group}: {len(info['blocks'])} blocks in {info['seconds']:.0f}s, peak {info.get('peak_gb', '?')} GB")
    spill = os.path.join(os.path.dirname(out_path) or ".", "_retrieval_spill")
    if "addrless_name_ngram_top5" in strategies:       # name 3-grams against the pool records without an address
        res = retrieve_in_processes(pre_dir, prefix, entity_ids, "addrless_ngram", 5, 0.3, n_threads, os.path.join(spill, "addrless"), log)
        add("addrless_name_ngram_top5", res.select(C.PAIR_COLS))
        del res
        gc.collect()
    if "addr_token_retrieval_top5" in strategies:      # ranks 1-5 of the top-20 run, exactly as it was measured on fold 0
        res = retrieve_in_processes(pre_dir, prefix, entity_ids, "addr", 20, 0.2, n_threads, os.path.join(spill, "addr"), log)
        add("addr_token_retrieval_top5", res.filter(pl.col("rank") <= 5).select(C.PAIR_COLS))
        del res
        gc.collect()
    if "name_addr_retrieval_top20" in strategies:
        res = retrieve_in_processes(pre_dir, prefix, entity_ids, "name_addr", k, 0.2, n_threads, os.path.join(spill, "name_addr"), log)
        add("name_addr_retrieval_top20", res.select(C.PAIR_COLS))
        del res
        gc.collect()
    shutil.rmtree(spill, ignore_errors=True)
    if keys_acc is None:
        keys_acc = np.zeros(0, dtype=np.uint64)
    q_arr, p_arr = C.unpack(keys_acc)
    final = C.pairs_frame(q_arr, p_arr, dedupe=False)
    del keys_acc, q_arr, p_arr
    gc.collect()
    vol = np.bincount(final["q_idx"].to_numpy(), minlength=n_s1) if final.height else np.zeros(n_s1, dtype=np.int64)
    stats["final"] = {"pairs": int(final.height), "mean": round(float(vol.mean()), 2), "median": float(np.median(vol)), "p95": float(np.percentile(vol, 95)),
                      "max": int(vol.max()), "entities_without_candidates": int((vol == 0).sum())}
    log(f"  union: {final.height:,} pairs, mean {vol.mean():.1f}, p95 {np.percentile(vol, 95):.0f}, max {vol.max():,}, "
        f"{int((vol == 0).sum()):,} entities without candidates")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pairs_out = os.path.splitext(out_path)[0] + ".parquet"        # positions (q_idx in the query order, p_idx in the S2+S3 order) for features.py
    final.select(C.PAIR_COLS).write_parquet(pairs_out)
    stats["pairs_parquet"] = pairs_out
    pids = pool_ids(files)
    C.write_candidate_pairs(final, s1_ids, pids, out_path)
    stats["file"] = C.validate_candidate_file(out_path, s1_ids, set(pids))
    stats["file"]["path"] = out_path
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"  wrote {out_path}: {stats['file']}")
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre-dir", required=True, help="directory with <prefix>_source{1,2,3}.parquet from preprocess.py")
    p.add_argument("--prefix", required=True, help="train or test")
    p.add_argument("--out", required=True, help="candidate_pairs.tsv to write")
    p.add_argument("--entity-ids", default=None, help="optional file with one Source 1 id per line (restricts the queries)")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--v1", action="store_true", help="use the seven-strategy set of candidate_generation.md instead of the final thirteen")
    a = p.parse_args(argv)
    ids = None
    if a.entity_ids:
        with open(a.entity_ids, encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    t0 = time.time()
    generate(a.pre_dir, a.prefix, a.out, ids, strategies=FINAL_STRATEGIES_V1 if a.v1 else FINAL_STRATEGIES, n_threads=a.procs, k=a.k,
             log=lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
