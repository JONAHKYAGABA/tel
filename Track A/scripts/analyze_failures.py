"""
scripts/analyze_failures.py

Side-by-side LLM vs heuristic vs ground-truth analysis for a labeled holdout.
Tells you WHERE the LLM under-performs the heuristic and what failure modes
each gets right.

Inputs:
  --llm_completions  eval/results/agentic_holdout/completions.jsonl
                     (or eval/results/holdout_lora_rag/completions.jsonl)
  --holdout          data/local_split/holdout_200.json   (labeled scenarios)

Outputs:
  • Confusion grid: {LLM right/wrong} × {heuristic right/wrong}
  • Per-failure-mode breakdown (COVERAGE / INTERFERENCE / SCHEDULER / UNKNOWN)
  • Single-vs-multi answer breakdown
  • IDs of scenarios where each approach beats the other (so you can inspect)
  • Cell-ID match rate (did LLM pick the right TEMPLATE but wrong CELL?)
  • Format-error rate (empty / malformed boxed answers)

Usage:
    python scripts/analyze_failures.py \\
        --llm_completions eval/results/agentic_holdout/completions.jsonl \\
        --holdout data/local_split/holdout_200.json \\
        --out_dir eval/analysis
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_baseline_submission import pick_answer as heuristic_pick  # noqa: E402
from scripts.build_baseline_submission import classify_mode as heuristic_classify  # noqa: E402
from scripts.build_baseline_submission import parse_table, find_drop  # noqa: E402


CX_RE = re.compile(r"^C\d+(\|C\d+)*$")


def iou(pred: str, gt: str) -> float:
    """IoU for multi-answer; exact for single."""
    if not pred or not gt:
        return 0.0
    p = set(pred.split("|"))
    g = set(gt.split("|"))
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def is_single(gt: str) -> bool:
    return "|" not in gt


def get_cells_in_options(scenario: Dict[str, Any]) -> Set[str]:
    """Extract the gNodeB_Cell IDs referenced in option labels."""
    cells: Set[str] = set()
    for opt in (scenario.get("task") or {}).get("options") or []:
        label = (opt.get("label") or "")
        # Look for patterns like "3279943_1" or "3267220_2"
        for m in re.finditer(r"\b(\d{6,8}_\d{1,2})\b", label):
            cells.add(m.group(1))
    return cells


def cell_ids_in_answer(scenario: Dict[str, Any], answer: str) -> Set[str]:
    """For a given answer (e.g. C7), find what cell IDs the picked options reference."""
    if not answer:
        return set()
    picked: Set[str] = set(answer.split("|"))
    cells: Set[str] = set()
    for opt in (scenario.get("task") or {}).get("options") or []:
        if opt.get("id") in picked:
            label = opt.get("label") or ""
            for m in re.finditer(r"\b(\d{6,8}_\d{1,2})\b", label):
                cells.add(m.group(1))
    return cells


def load_llm_completions(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_completions", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=PROJECT_DIR / "eval/analysis")
    ap.add_argument("--n_examples", type=int, default=10,
                    help="How many concrete examples to print per category.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    llm_recs = load_llm_completions(args.llm_completions)
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    print(f"[load] llm_completions={len(llm_recs)}  holdout={len(holdout)}")

    grid = {
        "both_correct":           [],  # both IoU == 1.0
        "both_wrong":             [],  # both IoU == 0
        "llm_right_heur_wrong":   [],  # llm IoU >= 0.5, heur IoU < 0.5
        "heur_right_llm_wrong":   [],  # heur IoU >= 0.5, llm IoU < 0.5
        "partial":                [],  # everything else
    }
    by_mode: Dict[str, Dict[str, List[float]]] = {}
    n_single = n_multi = 0
    llm_iou_sum = heur_iou_sum = 0.0
    n_format_errors = 0
    n_cell_mismatch = 0  # LLM picked option that references a cell never in the data drop window
    fallback_count = 0

    for s in holdout:
        sid = s.get("scenario_id")
        gt = s.get("answer", "")
        if not gt or gt == "To be determined":
            continue

        rec = llm_recs.get(sid, {})
        llm_ans = rec.get("answer", "")
        llm_source = rec.get("source", "missing")
        if llm_source == "heuristic":
            fallback_count += 1

        # Heuristic answer (re-compute fresh; not using the LLM record's value)
        heur_ans = heuristic_pick(s)

        # Failure-mode label (from heuristic's view of the data)
        diag = find_drop(parse_table(s.get("data", {}).get("user_plane_data", "")))
        mode = heuristic_classify(diag)

        llm_score = iou(llm_ans, gt)
        heur_score = iou(heur_ans, gt)

        llm_iou_sum += llm_score
        heur_iou_sum += heur_score

        if is_single(gt):
            n_single += 1
        else:
            n_multi += 1

        # Format errors
        if llm_ans and not CX_RE.match(llm_ans):
            n_format_errors += 1

        # Cell-ID mismatch (only meaningful for single-answer)
        if is_single(gt) and llm_ans:
            cells_picked = cell_ids_in_answer(s, llm_ans)
            cells_gt = cell_ids_in_answer(s, gt)
            if cells_gt and not (cells_picked & cells_gt):
                n_cell_mismatch += 1

        # Categorize
        if llm_score >= 1.0 and heur_score >= 1.0:
            grid["both_correct"].append((sid, mode, gt, llm_ans, heur_ans, llm_source))
        elif llm_score == 0.0 and heur_score == 0.0:
            grid["both_wrong"].append((sid, mode, gt, llm_ans, heur_ans, llm_source))
        elif llm_score >= 0.5 and heur_score < 0.5:
            grid["llm_right_heur_wrong"].append((sid, mode, gt, llm_ans, heur_ans, llm_source))
        elif heur_score >= 0.5 and llm_score < 0.5:
            grid["heur_right_llm_wrong"].append((sid, mode, gt, llm_ans, heur_ans, llm_source))
        else:
            grid["partial"].append((sid, mode, gt, llm_ans, heur_ans, llm_source))

        # Per-mode accumulator
        bucket = by_mode.setdefault(mode, {"llm": [], "heur": []})
        bucket["llm"].append(llm_score)
        bucket["heur"].append(heur_score)

    n_total = sum(len(v) for v in grid.values())

    # ---------- print summary ----------
    print()
    print("=" * 80)
    print(f"OVERALL  (n={n_total})")
    print("=" * 80)
    print(f"  LLM  mean IoU : {llm_iou_sum / max(n_total, 1):.4f}")
    print(f"  Heur mean IoU : {heur_iou_sum / max(n_total, 1):.4f}")
    print(f"  Single-answer : {n_single}    Multi-answer: {n_multi}")
    print(f"  LLM fell back to heuristic in this run: {fallback_count}")
    print(f"  LLM format errors: {n_format_errors}")
    print(f"  LLM cell-ID mismatch on single-answer: {n_cell_mismatch}")

    print()
    print("=" * 80)
    print("CONFUSION GRID (LLM × Heuristic)")
    print("=" * 80)
    for k, v in grid.items():
        pct = 100 * len(v) / max(n_total, 1)
        print(f"  {k:<26s}: {len(v):>4d}  ({pct:5.1f}%)")

    print()
    print("=" * 80)
    print("PER-FAILURE-MODE (heuristic's classification)")
    print("=" * 80)
    for mode, b in sorted(by_mode.items()):
        n = len(b["llm"])
        if n == 0:
            continue
        llm_m = sum(b["llm"]) / n
        heur_m = sum(b["heur"]) / n
        flag = "←  LLM wins" if llm_m > heur_m else ("←  Heur wins" if heur_m > llm_m else "")
        print(f"  {mode:<14s}  n={n:>4d}   LLM={llm_m:.3f}  Heur={heur_m:.3f}  {flag}")

    print()
    print("=" * 80)
    print("EXAMPLES — LLM RIGHT, HEUR WRONG  (look for prompt patterns to keep)")
    print("=" * 80)
    for sid, mode, gt, llm_ans, heur_ans, src in grid["llm_right_heur_wrong"][:args.n_examples]:
        print(f"  {sid[:8]} mode={mode:<14s} gt={gt:<18s} llm={llm_ans:<18s} heur={heur_ans:<18s} src={src}")

    print()
    print("=" * 80)
    print("EXAMPLES — HEUR RIGHT, LLM WRONG  (these are the prompt fixes to chase)")
    print("=" * 80)
    for sid, mode, gt, llm_ans, heur_ans, src in grid["heur_right_llm_wrong"][:args.n_examples]:
        print(f"  {sid[:8]} mode={mode:<14s} gt={gt:<18s} llm={llm_ans:<18s} heur={heur_ans:<18s} src={src}")

    print()
    print("=" * 80)
    print("EXAMPLES — BOTH WRONG  (genuinely hard scenarios)")
    print("=" * 80)
    for sid, mode, gt, llm_ans, heur_ans, src in grid["both_wrong"][:args.n_examples]:
        print(f"  {sid[:8]} mode={mode:<14s} gt={gt:<18s} llm={llm_ans:<18s} heur={heur_ans:<18s} src={src}")

    # ---------- write JSON summary ----------
    out_summary = {
        "n_total": n_total,
        "llm_mean_iou": round(llm_iou_sum / max(n_total, 1), 4),
        "heur_mean_iou": round(heur_iou_sum / max(n_total, 1), 4),
        "fallback_count": fallback_count,
        "format_errors": n_format_errors,
        "cell_id_mismatch_single": n_cell_mismatch,
        "by_mode": {
            mode: {
                "n": len(b["llm"]),
                "llm_iou": round(sum(b["llm"]) / max(len(b["llm"]), 1), 4),
                "heur_iou": round(sum(b["heur"]) / max(len(b["heur"]), 1), 4),
            }
            for mode, b in by_mode.items()
        },
        "categories": {k: [sid for sid, *_ in v] for k, v in grid.items()},
    }
    out_path = args.out_dir / "failure_analysis.json"
    out_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print()
    print(f"[write] {out_path}")

    # ---------- write per-scenario CSV for spreadsheet inspection ----------
    import csv
    csv_path = args.out_dir / "side_by_side.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "mode", "gt", "llm_ans", "heur_ans",
                    "llm_iou", "heur_iou", "category", "src"])
        for cat, rows in grid.items():
            for sid, mode, gt, llm_ans, heur_ans, src in rows:
                w.writerow([sid, mode, gt, llm_ans, heur_ans,
                            f"{iou(llm_ans, gt):.3f}", f"{iou(heur_ans, gt):.3f}",
                            cat, src])
    print(f"[write] {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
