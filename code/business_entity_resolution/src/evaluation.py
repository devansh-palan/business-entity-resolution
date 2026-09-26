#!/usr/bin/env python3
"""Local re-implementation of the ML Challenge 2026 leaderboard metric.

The problem statement defines the score as

    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

computed for every Source 1 entity and macro-averaged over *all* Source 1
entities of the evaluation set.  A singleton (an entity with no true matches)
scores 1.0 when the predicted list is empty and 0.0 when anything is predicted.

Points the statement leaves implicit, and how they are resolved here (each one
is pinned by a test in tests/test_evaluation.py):

* Comparison is set based: order is irrelevant and an ID counts once.
* Entity with true matches and an empty prediction: recall 0, F_0.5 = 0.
* Entity with true matches and no correct prediction: precision 0, F_0.5 = 0.
* Precision is reported as None when nothing was predicted and recall as None
  when there was nothing to find; F_0.5 is still defined by the rules above.
* The macro average divides by the number of entities in the evaluation set,
  never by the number of rows in the prediction file.
* A predicted ID that does not exist anywhere is simply a false positive
  (the statement says such IDs "only lower your score").

Strict mode (the default) also enforces the format rules of the official
validator, because the official scorer rejects such files: duplicate Source 1
rows, duplicate IDs inside a list, IDs that are not S2-/S3- prefixed, IDs with
surrounding whitespace, and prediction rows missing for, or absent from, the
evaluation set.  Lenient mode repairs those problems instead (dedupe, strip,
missing row = empty prediction, unknown row ignored) so that partial or messy
files can still be scored during development.

Files are read exactly the way utils/validate_submission.py reads them: text
mode with universal newlines, split on the first tab, IDs split on commas.

Command line:

    python src/evaluation.py --truth dataset/train/train_ground_truth.tsv \
        --pred output/matching_results.tsv [--ids ids.txt] [--lenient] [--by-k]
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BETA = 0.5
BETA_SQ = BETA * BETA
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
VALID_PREFIXES = ("S2-", "S3-")


class SubmissionFormatError(ValueError):
    """Raised in strict mode for anything the official validator would reject."""


# --------------------------------------------------------------------------- #
# Entity level
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EntityScore:
    entity_id: str
    n_true: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: Optional[float]  # None when nothing was predicted
    recall: Optional[float]     # None when there was nothing to find
    f05: float


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """F_beta exactly as written in the problem statement. Returns 0 when both are 0."""
    b2 = beta * beta
    denominator = b2 * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denominator


def _as_id_set(ids: Iterable[str], label: str) -> set:
    if isinstance(ids, str):
        raise TypeError(f"{label}: expected an iterable of IDs, got a bare string {ids!r}")
    return set(ids)


def score_entity(entity_id: str, true_ids: Iterable[str], pred_ids: Iterable[str]) -> EntityScore:
    """Precision, recall and F_0.5 for one Source 1 entity."""
    truth = _as_id_set(true_ids, "true_ids")
    pred = _as_id_set(pred_ids, "pred_ids")
    tp = len(truth & pred)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / len(pred) if pred else None
    recall = tp / len(truth) if truth else None
    if not truth:                 # singleton: empty prediction is worth 1, anything else 0
        f05 = 1.0 if not pred else 0.0
    elif tp == 0:                 # nothing right: empty prediction or only wrong IDs
        f05 = 0.0
    else:
        f05 = f_beta(precision, recall)
    return EntityScore(entity_id, len(truth), len(pred), tp, fp, fn, precision, recall, f05)


# --------------------------------------------------------------------------- #
# Macro average
# --------------------------------------------------------------------------- #
@dataclass
class MacroResult:
    macro_f05: float
    n_entities: int
    n_singletons: int
    singleton_f05: float        # mean F over singletons = share predicted empty
    non_singleton_f05: float    # mean F over entities that have matches
    by_n_true: Dict[int, Tuple[int, float]]  # k -> (number of entities, mean F)
    pooled_precision: Optional[float]         # micro averages, informational only
    pooled_recall: Optional[float]
    n_pred_nonempty: int
    per_entity: Optional[List[EntityScore]] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"macro F0.5           : {self.macro_f05:.6f}",
            f"entities scored      : {self.n_entities:,}",
            f"singletons           : {self.n_singletons:,}  (mean F = {self.singleton_f05:.4f})",
            f"entities with matches: {self.n_entities - self.n_singletons:,}  (mean F = {self.non_singleton_f05:.4f})",
            f"non-empty predictions: {self.n_pred_nonempty:,}",
        ]
        if self.pooled_precision is not None and self.pooled_recall is not None:
            lines.append(f"pooled P / R (info)  : {self.pooled_precision:.4f} / {self.pooled_recall:.4f}")
        return "\n".join(lines)

    def by_k_table(self) -> str:
        rows = ["  k   entities   mean F0.5"]
        for k in sorted(self.by_n_true):
            n, f = self.by_n_true[k]
            rows.append(f"{k:>3} {n:>10,}   {f:.4f}")
        return "\n".join(rows)


def macro_f05(
    truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    entity_ids: Optional[Iterable[str]] = None,
    strict: bool = True,
    keep_per_entity: bool = False,
) -> MacroResult:
    """Macro-averaged F_0.5 over the evaluation set.

    truth        : entity id -> its true matched ids (the keys define the evaluation set)
    predictions  : entity id -> predicted matched ids
    entity_ids   : optional subset of the evaluation set to score (e.g. a validation fold)
    strict       : True  -> every scored entity must have a prediction row and, when the
                            whole set is scored, no unknown rows may be present
                   False -> missing rows count as empty predictions, unknown rows are ignored
    """
    if entity_ids is None:
        ids: List[str] = list(truth.keys())
    else:
        ids = list(entity_ids)
        unknown = [e for e in ids if e not in truth]
        if unknown:
            raise KeyError(f"{len(unknown)} requested entities are not in the truth, e.g. {unknown[:5]}")
    if strict:
        missing = [e for e in ids if e not in predictions]
        if missing:
            raise SubmissionFormatError(
                f"{len(missing)} entities have no prediction row, e.g. {sorted(missing)[:5]}"
            )
        if entity_ids is None:
            extra = [e for e in predictions if e not in truth]
            if extra:
                raise SubmissionFormatError(
                    f"{len(extra)} prediction rows refer to unknown entities, e.g. {sorted(extra)[:5]}"
                )

    n = len(ids)
    if n == 0:
        raise ValueError("no entities to score")
    # Per-entity scores are collected and summed with math.fsum (exactly rounded), so the
    # macro average does not drift with the accumulation order over millions of entities.
    f_by_k: Dict[int, List[float]] = {}
    n_pred_nonempty = 0
    tp_sum = fp_sum = fn_sum = 0
    per_entity: List[EntityScore] = []
    empty: Tuple[str, ...] = ()
    for e in ids:
        s = score_entity(e, truth[e], predictions.get(e, empty))
        tp_sum += s.tp
        fp_sum += s.fp
        fn_sum += s.fn
        if s.n_pred:
            n_pred_nonempty += 1
        f_by_k.setdefault(s.n_true, []).append(s.f05)
        if keep_per_entity:
            per_entity.append(s)
    total_by_k = {k: math.fsum(v) for k, v in f_by_k.items()}
    total = math.fsum(total_by_k.values())
    singleton_scores = f_by_k.get(0, [])
    n_singletons = len(singleton_scores)
    singleton_total = total_by_k.get(0, 0.0)
    n_non = n - n_singletons
    return MacroResult(
        macro_f05=total / n,
        n_entities=n,
        n_singletons=n_singletons,
        singleton_f05=(singleton_total / n_singletons) if n_singletons else float("nan"),
        non_singleton_f05=((total - singleton_total) / n_non) if n_non else float("nan"),
        by_n_true={k: (len(v), total_by_k[k] / len(v)) for k, v in f_by_k.items()},
        pooled_precision=(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) else None,
        pooled_recall=(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) else None,
        n_pred_nonempty=n_pred_nonempty,
        per_entity=per_entity if keep_per_entity else None,
    )


def macro_f05_from_rows(
    truth: Mapping[str, Iterable[str]],
    rows: Iterable[Tuple[str, Iterable[str]]],
    keep_per_entity: bool = False,
) -> MacroResult:
    """macro_f05 for predictions streamed as (entity_id, predicted ids) rows instead of a dict.

    Rows cover the entities with a non-empty prediction; every other entity of ``truth`` is scored as an
    empty prediction.  Rows for unknown entities are ignored, a repeated entity id is an error.  Scoring and
    aggregation are exactly those of macro_f05 (score_entity per entity, math.fsum), so a prediction table of
    millions of pairs can be scored without building a Python dict of every id.
    """
    f_by_k: Dict[int, List[float]] = {}
    n_pred_nonempty = 0
    tp_sum = fp_sum = fn_sum = 0
    per_entity: List[EntityScore] = []
    seen: set = set()

    def take(s: EntityScore) -> None:
        nonlocal n_pred_nonempty, tp_sum, fp_sum, fn_sum
        tp_sum += s.tp
        fp_sum += s.fp
        fn_sum += s.fn
        if s.n_pred:
            n_pred_nonempty += 1
        f_by_k.setdefault(s.n_true, []).append(s.f05)
        if keep_per_entity:
            per_entity.append(s)

    for e, pred in rows:
        if e not in truth:
            continue
        if e in seen:
            raise ValueError(f"entity {e} appears in more than one prediction row")
        seen.add(e)
        take(score_entity(e, truth[e], pred))
    empty: Tuple[str, ...] = ()
    for e, t in truth.items():
        if e not in seen:
            take(score_entity(e, t, empty))
    n = len(truth)
    if n == 0:
        raise ValueError("no entities to score")
    total_by_k = {k: math.fsum(v) for k, v in f_by_k.items()}
    total = math.fsum(total_by_k.values())
    singleton_scores = f_by_k.get(0, [])
    n_singletons = len(singleton_scores)
    singleton_total = total_by_k.get(0, 0.0)
    n_non = n - n_singletons
    return MacroResult(
        macro_f05=total / n,
        n_entities=n,
        n_singletons=n_singletons,
        singleton_f05=(singleton_total / n_singletons) if n_singletons else float("nan"),
        non_singleton_f05=((total - singleton_total) / n_non) if n_non else float("nan"),
        by_n_true={k: (len(v), total_by_k[k] / len(v)) for k, v in f_by_k.items()},
        pooled_precision=(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) else None,
        pooled_recall=(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) else None,
        n_pred_nonempty=n_pred_nonempty,
        per_entity=per_entity if keep_per_entity else None,
    )


# --------------------------------------------------------------------------- #
# File format (mirrors utils/validate_submission.py)
# --------------------------------------------------------------------------- #
def parse_id_list(raw: str, strict: bool = True, where: str = "") -> List[str]:
    """Parse the comma separated ID list of one row.

    Strict mode reproduces the validator's rejections: an empty element or
    whitespace around an element ("S2-1, S2-2" fails its prefix check), a
    non S2-/S3- prefix, or a repeated ID.  Lenient mode strips and dedupes.
    """
    if raw.strip() == "":
        return []
    ids = raw.split(",")
    if not strict:
        cleaned = [i.strip() for i in ids]
        return list(dict.fromkeys(i for i in cleaned if i))
    for i in ids:
        if i == "" or i != i.strip():
            raise SubmissionFormatError(
                f"{where}: empty element or whitespace around an ID in {raw!r} "
                "(the validator rejects IDs that do not start with S2-/S3-)"
            )
        if not i.startswith(VALID_PREFIXES):
            raise SubmissionFormatError(f"{where}: ID {i!r} is not S2-/S3- prefixed")
    if len(set(ids)) != len(ids):
        raise SubmissionFormatError(f"{where}: duplicate IDs inside the list {raw!r}")
    return ids


def read_id_list_file(
    path: str,
    expected_header: Sequence[str] = MATCHING_HEADER,
    strict: bool = True,
) -> Dict[str, List[str]]:
    """Read a ground truth / matching_results / candidate_pairs TSV into a dict."""
    mapping: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:  # universal newlines, same as the validator
        header = f.readline()
        if not header:
            raise SubmissionFormatError(f"{path}: file is empty")
        if "\t" not in header and "," in header:
            raise SubmissionFormatError(f"{path}: header has no tab but has commas; this is a CSV, not a TSV")
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
        if cols != [c.lower() for c in expected_header]:
            raise SubmissionFormatError(f"{path}: header {cols} but expected {list(expected_header)}")
        for line_no, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            if not tab:
                if s1.strip() == "":
                    continue  # blank line, ignored by the validator as well
                raise SubmissionFormatError(f"{path}:{line_no}: row has no tab: {line.rstrip()!r}")
            if s1 != s1.strip():
                if strict:
                    raise SubmissionFormatError(f"{path}:{line_no}: whitespace around the entity id {s1!r}")
                s1 = s1.strip()
            if s1 in mapping:
                if strict:
                    raise SubmissionFormatError(f"{path}:{line_no}: duplicate row for {s1}")
                continue  # lenient: first row wins
            mapping[s1] = parse_id_list(rest.rstrip("\n"), strict, f"{path}:{line_no}")
    return mapping


def write_id_list_file(
    path: str,
    mapping: Mapping[str, Iterable[str]],
    header: Sequence[str] = MATCHING_HEADER,
    entity_order: Optional[Iterable[str]] = None,
    validate: bool = True,
) -> None:
    """Write a matching_results / candidate_pairs TSV: UTF-8, LF line endings, no quoting.

    entity_order lets the caller emit one row per required Source 1 entity, in the
    order of the test file; entities absent from mapping get an empty list.
    """
    order = list(entity_order) if entity_order is not None else list(mapping.keys())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for e in order:
            ids = list(mapping.get(e, ()))
            if validate:
                if "\t" in e or e != e.strip():
                    raise SubmissionFormatError(f"bad entity id {e!r}")
                for i in ids:
                    if i != i.strip() or "," in i or "\t" in i or not i.startswith(VALID_PREFIXES):
                        raise SubmissionFormatError(f"{e}: bad matched id {i!r}")
                if len(set(ids)) != len(ids):
                    raise SubmissionFormatError(f"{e}: duplicate ids {ids}")
            f.write(f"{e}\t{','.join(ids)}\n")


def read_entity_ids(path: str) -> List[str]:
    """One entity id per line, or the first column of a TSV (header row skipped if it is not an id)."""
    out: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split("\t", 1)[0].strip()
            if tok and tok != "entity_id" and tok != "source1_entity_id":
                out.append(tok)
    return out


def score_files(
    truth_path: str,
    pred_path: str,
    ids_path: Optional[str] = None,
    strict: bool = True,
    keep_per_entity: bool = False,
) -> MacroResult:
    truth = read_id_list_file(truth_path, MATCHING_HEADER, strict=True)
    preds = read_id_list_file(pred_path, MATCHING_HEADER, strict=strict)
    ids = read_entity_ids(ids_path) if ids_path else None
    return macro_f05(truth, preds, entity_ids=ids, strict=strict, keep_per_entity=keep_per_entity)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Local macro F0.5 scorer for the entity resolution challenge.")
    p.add_argument("--truth", required=True, help="ground truth TSV (source1_entity_id, matched_entity_ids)")
    p.add_argument("--pred", required=True, help="predictions TSV in matching_results.tsv format")
    p.add_argument("--ids", default=None, help="optional file listing the Source 1 ids to score (a fold)")
    p.add_argument("--lenient", action="store_true", help="repair format problems instead of failing")
    p.add_argument("--by-k", action="store_true", help="also print mean F0.5 by number of true matches")
    a = p.parse_args(argv)
    try:
        res = score_files(a.truth, a.pred, a.ids, strict=not a.lenient)
    except SubmissionFormatError as exc:
        print(f"FORMAT ERROR: {exc}", file=sys.stderr)
        return 2
    print(res.summary())
    if a.by_k:
        print(res.by_k_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
