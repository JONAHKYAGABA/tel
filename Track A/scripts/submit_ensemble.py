"""
scripts/submit_ensemble.py

Confidence-routing ensemble between LLM and heuristic. Produces a NEW result.csv
from existing LLM completions + on-the-fly heuristic, plus rule-based confidence.

Rules (in priority order):
  1. If LLM and heuristic AGREE on the answer  → use that answer (highest confidence)
  2. If LLM answer is empty / malformed         → use heuristic
  3. If LLM source was already 'heuristic'      → use heuristic (LLM never ran)
  4. If the scenario is MULTI-answer and LLM picked only ONE option
     while the heuristic picked 2-3                → use heuristic's multi pick
     (multi tasks scored by IoU; missing a correct option costs as much
      as adding a wrong one, so recall > precision)
  5. Otherwise (both produced valid disagreeing answers) → use LLM
     (LLM has more information; only override when we have a reason)

Two CSVs are written so you can also try a "LLM-leaning" tiebreak vs the
default "heuristic-leaning" tiebreak.

Usage:
    python scripts/submit_ensemble.py \\
        --llm_completions eval/results/agentic_holdout/completions.jsonl \\
        --test_file       data/local_split/holdout_200.json \\
        --out_dir         eval/results/ensemble_holdout

Also works with the final Phase-2 completions:
    python scripts/submit_ensemble.py \\
        --llm_completions eval/results/final_baseline/completions.jsonl \\
        --test_file       data/Phase_2/test.json \\
        --out_dir         eval/results/ensemble_final
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_baseline_submission import (  # noqa: E402
    pick_answer as heuristic_pick,
    is_multi as task_is_multi,
)


CX_RE = re.compile(r"^C\d+(\|C\d+)*$")


def normalize(answer: str) -> str:
    if not answer:
        return ""
    parts = [p.strip() for p in answer.split("|") if p.strip()]
    def key(s: str) -> int:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0
    parts = sorted(set(parts), key=key)
    return "|".join(parts)


def iou_score(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    p = set(pred.split("|"))
    g = set(gt.split("|"))
    return len(p & g) / max(len(p | g), 1) if (p or g) else 0.0


def load_llm_completions(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                if sid:
                    out[sid] = rec
            except json.JSONDecodeError:
                continue
    return out


def ensemble_one(scenario: Dict[str, Any], llm_rec: Dict[str, Any],
                 tiebreak: str) -> Dict[str, str]:
    """Apply the confidence rules. Returns (answer, source, reason)."""
    sid = scenario.get("scenario_id", "")
    llm_ans_raw = (llm_rec or {}).get("answer", "") or ""
    llm_ans = normalize(llm_ans_raw)
    llm_source = (llm_rec or {}).get("source", "missing")
    is_multi = task_is_multi(scenario)

    heur_ans = normalize(heuristic_pick(scenario))

    # Rule 1: agreement
    if llm_ans and llm_ans == heur_ans:
        return {"answer": llm_ans, "source": "both_agree", "reason": "exact match"}

    # Rule 2: malformed LLM answer
    if not llm_ans or not CX_RE.match(llm_ans):
        return {"answer": heur_ans, "source": "heuristic",
                "reason": f"llm_malformed: {llm_ans_raw!r}"}

    # Rule 3: LLM didn't actually run
    if llm_source in ("heuristic", "missing"):
        return {"answer": heur_ans, "source": "heuristic",
                "reason": f"llm_never_ran (src={llm_source})"}

    # Rule 4: multi-answer task but LLM emitted only one option
    if is_multi and "|" not in llm_ans:
        if heur_ans and "|" in heur_ans:
            return {"answer": heur_ans, "source": "heuristic",
                    "reason": "multi_task_llm_single_only"}
        # Last-ditch: merge LLM's one option with heuristic's options
        if heur_ans:
            merged = normalize("|".join({llm_ans, heur_ans}))
            return {"answer": merged, "source": "merged",
                    "reason": "multi_merged_with_heuristic"}

    # Rule 5: both valid + disagree → tiebreak
    if tiebreak == "llm":
        return {"answer": llm_ans, "source": "llm",
                "reason": "tiebreak=llm"}
    else:
        return {"answer": heur_ans, "source": "heuristic",
                "reason": "tiebreak=heuristic"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_completions", type=Path, required=True)
    ap.add_argument("--test_file", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--score_against_gt", action="store_true",
                    help="If test_file has ground-truth answers, print scores.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    llm_recs = load_llm_completions(args.llm_completions)
    scenarios = json.loads(args.test_file.read_text(encoding="utf-8"))
    print(f"[load] llm_completions={len(llm_recs)}  scenarios={len(scenarios)}")

    out: Dict[str, List[Dict[str, str]]] = {"heur_tiebreak": [], "llm_tiebreak": []}
    decisions = []
    for s in scenarios:
        sid = s.get("scenario_id", "")
        rec = llm_recs.get(sid, {})
        for tiebreak in ("heur", "llm"):
            d = ensemble_one(s, rec, tiebreak)
            out[f"{tiebreak}_tiebreak"].append({"scenario_id": sid, "answers": d["answer"]})
            if tiebreak == "heur":
                decisions.append({
                    "scenario_id": sid,
                    "answer": d["answer"],
                    "source": d["source"],
                    "reason": d["reason"],
                    "llm_raw": rec.get("answer", ""),
                    "is_multi": task_is_multi(s),
                })

    # Source breakdown
    src_counts: Dict[str, int] = {}
    for d in decisions:
        src_counts[d["source"]] = src_counts.get(d["source"], 0) + 1
    print()
    print("Decision breakdown (heur tiebreak):")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20s}: {v:>4d}  ({100*v/max(len(decisions),1):.1f}%)")

    # Score if ground truth is present
    labeled = [s for s in scenarios if s.get("answer") and s.get("answer") != "To be determined"]
    if labeled or args.score_against_gt:
        gt_by_id = {s["scenario_id"]: s.get("answer", "") for s in labeled}
        if gt_by_id:
            for tiebreak, rows in out.items():
                ans_by_id = {r["scenario_id"]: r["answers"] for r in rows}
                scores = []
                for sid, gt in gt_by_id.items():
                    pred = ans_by_id.get(sid, "")
                    scores.append(iou_score(pred, gt))
                mean = sum(scores) / len(scores)
                print(f"  ENSEMBLE [{tiebreak:>5s}_tiebreak]  mean IoU = {mean:.4f}  (n={len(scores)})")

    # Write the two ensemble CSVs in Zindi format
    for tiebreak, rows in out.items():
        df = pd.DataFrame(rows, dtype=str).fillna("")
        df = df.rename(columns={"scenario_id": "ID", "answers": "Track A"})
        df["Track B"] = ""
        df = df[["ID", "Track A", "Track B"]]
        outfile = args.out_dir / f"result_ensemble_{tiebreak}_zindi.csv"
        df.to_csv(outfile, index=False)
        print(f"[write] {outfile}  ({len(df)} rows)")

    # Also write decisions log for debugging
    log_path = args.out_dir / "decisions.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[write] {log_path}  ({len(decisions)} decisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
