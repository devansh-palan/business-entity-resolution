"""Apply the name normaliser (names.py) and the address parser (addresses.py) to every record of a split.

The result is one parquet dataset per source file with flat, join-friendly columns: the original text, every
name representation, every parsed address component and the flags of both parsers.  Deterministic rules,
candidate generation and the feature builder all read these tables instead of re-parsing text.

Fold safety: the transliteration dictionary and the address vocabulary are learned from training pairs, so an
evaluation on fold k must use the dictionaries built without fold k (artifacts/validation/*_excl_fold0.json).
The test-time run uses the dictionaries learned from all training pairs.

Usage (from the project root):
    python code/business_entity_resolution/src/preprocess.py --split-dir student_resource/dataset/train \
        --prefix train --out-dir artifacts/preprocessed/fold0 \
        --translit artifacts/validation/translit_dict_excl_fold0.json \
        --vocab artifacts/validation/address_vocab_excl_fold0.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import addresses as ad  # noqa: E402
import names as nm  # noqa: E402

SCHEMA: Dict[str, pl.DataType] = {
    "entity_id": pl.String, "src": pl.String, "country": pl.String, "business_name": pl.String, "business_address": pl.String,
    # name representations
    "name_clean": pl.String, "name_core": pl.String, "name_core_sorted": pl.String, "name_compact": pl.String,
    "name_acronym": pl.String, "name_translit_core": pl.String, "name_translit_core_sorted": pl.String,
    "name_tokens": pl.List(pl.String), "name_core_tokens": pl.List(pl.String), "name_legal_forms": pl.List(pl.String),
    "name_alias_prefix": pl.String, "name_honorific": pl.String, "name_variants": pl.List(pl.String),
    "name_digits": pl.List(pl.String), "name_script": pl.String, "name_flags": pl.List(pl.String),
    # parsed address
    "addr_norm": pl.String, "addr_sorted": pl.String, "house_number": pl.String, "street": pl.String, "street_type": pl.String,
    "unit": pl.String, "city": pl.String, "city_norm": pl.String, "state": pl.String, "state_recognised": pl.Boolean,
    "postal_code": pl.String, "landmark": pl.List(pl.String), "locality": pl.List(pl.String), "po_box": pl.String,
    "care_of": pl.String, "addr_numbers": pl.List(pl.String), "addr_digit_runs": pl.List(pl.String),
    "addr_tokens": pl.List(pl.String), "addr_components": pl.List(pl.String), "addr_missing": pl.Boolean,
    "addr_flags": pl.List(pl.String),
}

_TRANSLIT: Optional[Dict[str, str]] = None
_VOCAB: Optional[ad.AddressVocab] = None


def _init(translit_path: Optional[str], vocab_path: Optional[str]) -> None:
    global _TRANSLIT, _VOCAB
    _TRANSLIT = nm.load_translit_dict(translit_path) if translit_path else None
    _VOCAB = ad.AddressVocab.load(vocab_path) if vocab_path else None


def _none_if_empty(s: Optional[str]) -> Optional[str]:
    return s if s else None


def process_records(ids: Sequence[str], names: Sequence[Optional[str]], addrs: Sequence[Optional[str]],
                    countries: Sequence[str], src: str) -> Dict[str, list]:
    """Normalise one chunk of records.  Returns a dict of columns following SCHEMA."""
    cols: Dict[str, list] = {k: [] for k in SCHEMA}
    for eid, name, addr, country in zip(ids, names, addrs, countries):
        n = nm.normalize_name(name or "", _TRANSLIT)
        a = ad.parse_address(addr, country, _VOCAB)
        cols["entity_id"].append(eid); cols["src"].append(src); cols["country"].append(country)
        cols["business_name"].append(name); cols["business_address"].append(addr)
        cols["name_clean"].append(_none_if_empty(n.clean)); cols["name_core"].append(_none_if_empty(n.core))
        cols["name_core_sorted"].append(_none_if_empty(n.core_sorted)); cols["name_compact"].append(_none_if_empty(n.compact))
        cols["name_acronym"].append(_none_if_empty(n.acronym)); cols["name_translit_core"].append(_none_if_empty(n.translit_core))
        cols["name_translit_core_sorted"].append(_none_if_empty(n.translit_core_sorted))
        cols["name_tokens"].append(n.tokens); cols["name_core_tokens"].append(n.core_tokens); cols["name_legal_forms"].append(n.legal_forms)
        cols["name_alias_prefix"].append(n.alias_prefix); cols["name_honorific"].append(n.honorific)
        cols["name_variants"].append(n.variants); cols["name_digits"].append(n.digits); cols["name_script"].append(n.script)
        cols["name_flags"].append(sorted(k for k, v in n.flags.items() if v))
        cols["addr_norm"].append(_none_if_empty(" ".join(a.tokens))); cols["addr_sorted"].append(_none_if_empty(" ".join(sorted(a.tokens))))
        cols["house_number"].append(a.house_number); cols["street"].append(a.street); cols["street_type"].append(a.street_type)
        cols["unit"].append(a.unit); cols["city"].append(a.city); cols["city_norm"].append(_none_if_empty(a.city_norm))
        cols["state"].append(a.state); cols["state_recognised"].append(bool(a.state_recognised)); cols["postal_code"].append(a.postal_code)
        cols["landmark"].append(a.landmark); cols["locality"].append(a.locality); cols["po_box"].append(a.po_box); cols["care_of"].append(a.care_of)
        cols["addr_numbers"].append(a.numbers); cols["addr_digit_runs"].append(a.digit_runs); cols["addr_tokens"].append(a.tokens)
        cols["addr_components"].append(a.components); cols["addr_missing"].append(bool(a.missing))
        cols["addr_flags"].append(sorted(k for k, v in a.flags.items() if v))
    return cols


def _work(args) -> Dict[str, list]:
    return process_records(*args)


def _work_to_file(args, part_path: str) -> int:
    """Worker entry point: normalise a chunk and write it straight to a parquet part (no result transfer)."""
    cols = process_records(*args)
    pl.DataFrame(cols, schema=SCHEMA).write_parquet(part_path, compression="zstd")
    return len(cols["entity_id"])


def _read_source(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="	", quote_char=None, has_header=True, infer_schema_length=0, encoding="utf8")


def preprocess_file(path: str, src: str, out_path: str, translit_path: Optional[str], vocab_path: Optional[str],
                    n_procs: int = 8, chunk: int = 40_000, log=print) -> int:
    """Normalise one source file into ``out_path`` (parquet).  Returns the number of rows written.

    Each worker writes its chunk as a parquet part; the parent only slices the input (at most 2 * n_procs
    chunks in flight) and finally concatenates the parts in order with a streaming sink.
    """
    import glob
    import shutil
    from concurrent.futures import FIRST_COMPLETED, wait

    df = _read_source(path)
    n = df.height
    part_dir = out_path + ".parts"
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir, exist_ok=True)

    def task(i: int):
        sl = df.slice(i, chunk)
        return (sl["entity_id"].to_list(), sl["business_name"].to_list(), sl["business_address"].to_list(), sl["country"].to_list(), src)

    starts = list(range(0, n, chunk))
    written = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_procs, initializer=_init, initargs=(translit_path, vocab_path)) as ex:
        pending = {}
        next_i = 0
        window = 2 * n_procs
        done_chunks = 0
        while next_i < len(starts) or pending:
            while next_i < len(starts) and len(pending) < window:
                part_path = os.path.join(part_dir, f"part-{next_i:05d}.parquet")
                fut = ex.submit(_work_to_file, task(starts[next_i]), part_path)
                pending[fut] = next_i
                next_i += 1
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                written += fut.result()
                done_chunks += 1
                if done_chunks % 25 == 0:
                    log(f"  {src}: {written:,}/{n:,} rows, {written / (time.time() - t0):,.0f} rows/s")
    del df
    parts = sorted(glob.glob(os.path.join(part_dir, "part-*.parquet")))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if parts:
        pl.scan_parquet(parts).sink_parquet(out_path, compression="zstd")
    else:
        pl.DataFrame(schema=SCHEMA).write_parquet(out_path, compression="zstd")
    shutil.rmtree(part_dir, ignore_errors=True)
    return written


def preprocess_split(split_dir: str, prefix: str, out_dir: str, translit_path: Optional[str], vocab_path: Optional[str],
                     sources: Sequence[int] = (1, 2, 3), n_procs: int = 8, chunk: int = 40_000, log=print) -> Dict[str, int]:
    log = lambda *a, _l=log: _l(*a, flush=True) if _l is print else _l(*a)
    counts = {}
    for k in sources:
        path = os.path.join(split_dir, f"{prefix}_source{k}.tsv")
        out_path = os.path.join(out_dir, f"{prefix}_source{k}.parquet")
        t0 = time.time()
        counts[f"source{k}"] = preprocess_file(path, f"S{k}", out_path, translit_path, vocab_path, n_procs, chunk, log=log)
        log(f"{prefix}_source{k}: {counts[f'source{k}']:,} rows in {time.time() - t0:,.0f}s -> {out_path}")
    return counts


def load_split(out_dir: str, prefix: str, sources: Sequence[int] = (1, 2, 3), columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(os.path.join(out_dir, f"{prefix}_source{k}.parquet"), columns=columns) for k in sources])


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--prefix", required=True, help="train or test")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--translit", default=None, help="transliteration dictionary json")
    p.add_argument("--vocab", default=None, help="address vocabulary json")
    p.add_argument("--sources", default="1,2,3")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--chunk", type=int, default=40_000)
    a = p.parse_args(argv)
    preprocess_split(a.split_dir, a.prefix, a.out_dir, a.translit, a.vocab, [int(x) for x in a.sources.split(",")], a.procs, a.chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
