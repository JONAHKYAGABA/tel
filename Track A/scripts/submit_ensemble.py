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
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_baseline_submission import (  # noqa: E402
    pick_answer as heuristic_pick,
    is_multi as task_is_multi,
    heuristic_diagnosis,
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


# ----------------------------- hybrid fusion -----------------------------
#
# ENSEMBLE_HYBRID=1 path: confidence-gated four-source weighted voting.
# Sources: agentic LLM, k-NN voting, direct LoRA prediction, heuristic.
# If agent's effective_confidence ≥ trust_threshold AND it called ≥ min_tools,
# use the agent's answer directly. Otherwise compute weighted votes per option
# and pick the top-k (k = 1 single / 2-3 multi).

def _knn_load_or_build(train_path: Path, cache_path: Path) -> bool:
    """Lazy-load the k-NN cache by reusing agentic_agent._knn_init.
    Returns True if loaded successfully."""
    try:
        from scripts.agentic_agent import _knn_init  # noqa: E402
        return _knn_init(str(train_path), str(cache_path))
    except Exception as exc:
        print(f"[hybrid] failed to init k-NN: {exc}", file=sys.stderr)
        return False


def _knn_vote(scenario: Dict[str, Any], k: int = 10) -> Tuple[str, float]:
    """Weighted (similarity²) vote over top-k training scenarios.
    Returns (answer, vote_share_for_winner)."""
    try:
        from scripts.agentic_agent import _knn_retrieve  # noqa: E402
    except Exception:
        return "", 0.0
    sim = _knn_retrieve(scenario, k=k)
    if not sim:
        return "", 0.0
    votes: Dict[str, float] = {}
    total = 0.0
    for ex in sim:
        ans = normalize(ex.get("answer", ""))
        if not ans:
            continue
        w = float(ex.get("sim", 0.0)) ** 2
        votes[ans] = votes.get(ans, 0.0) + w
        total += w
    if not votes or total <= 0:
        return "", 0.0
    winner = max(votes.items(), key=lambda kv: kv[1])
    return winner[0], winner[1] / total


def _option_ids(scenario: Dict[str, Any]) -> List[str]:
    return [o.get("id", "") for o in (scenario.get("task") or {}).get("options") or [] if o.get("id")]


def _split_pipe(ans: str) -> List[str]:
    return [p for p in (ans or "").split("|") if p]


def _top_k_from_votes(option_scores: Dict[str, float], k: int) -> str:
    """Pick top-k option IDs by score, return pipe-separated Cx in ascending numeric order."""
    if not option_scores:
        return ""
    sorted_items = sorted(option_scores.items(), key=lambda kv: -kv[1])
    picked = [opt for opt, sc in sorted_items[:k] if sc > 0]
    if not picked:
        picked = [sorted_items[0][0]]
    return normalize("|".join(picked))


def hybrid_fuse_one(scenario: Dict[str, Any],
                    agent_rec: Dict[str, Any],
                    lora_rec: Optional[Dict[str, Any]],
                    weights: Dict[str, float],
                    trust_threshold: float = 0.75,
                    min_tools: int = 2,
                    knn_k: int = 10) -> Dict[str, Any]:
    """Confidence-gated four-source fusion. Returns the same shape as
    ensemble_one plus diagnostic fields."""
    sid = scenario.get("scenario_id", "")
    multi = task_is_multi(scenario)
    valid = set(_option_ids(scenario))

    # Agent
    agent_ans = normalize((agent_rec or {}).get("answer", "") or "")
    agent_conf = float((agent_rec or {}).get("confidence", 0.0) or 0.0)
    agent_eff = float((agent_rec or {}).get("effective_confidence", agent_conf) or agent_conf)
    agent_tools = int((agent_rec or {}).get("num_tool_calls", 0) or 0)
    agent_src = (agent_rec or {}).get("source", "missing")

    # Heuristic
    heur_diag = heuristic_diagnosis(scenario)
    heur_ans = normalize(heur_diag.get("recommended_answer", "") or "")
    heur_conf = float(heur_diag.get("confidence", 0.5))

    # k-NN
    knn_ans, knn_conf = _knn_vote(scenario, k=knn_k)
    knn_ans = normalize(knn_ans)

    # LoRA (optional)
    lora_ans = normalize((lora_rec or {}).get("answer", "") or "") if lora_rec else ""

    # Gate: if agent is confident AND used enough tools AND answer is valid, trust it.
    agent_valid = bool(agent_ans) and CX_RE.match(agent_ans) is not None
    if (
        agent_valid and agent_src not in ("heuristic", "missing")
        and agent_eff >= trust_threshold
        and agent_tools >= min_tools
    ):
        return {
            "answer": agent_ans,
            "source": "agent_trusted",
            "reason": f"conf={agent_eff:.2f}≥{trust_threshold} tools={agent_tools}",
            "agent": agent_ans, "knn": knn_ans, "lora": lora_ans, "heur": heur_ans,
            "agent_conf": agent_eff, "knn_conf": knn_conf, "heur_conf": heur_conf,
        }

    # Weighted vote — score each individual option (Cx), not whole strings,
    # so multi-answer choices can borrow components from different sources.
    option_scores: Dict[str, float] = {opt: 0.0 for opt in valid}

    def _add(votes: List[str], weight: float) -> None:
        if not votes or weight <= 0:
            return
        for v in votes:
            if v in option_scores:
                option_scores[v] += weight

    if agent_valid:
        _add(_split_pipe(agent_ans), weights.get("agent", 0.4) * max(agent_eff, 0.1))
    if knn_ans:
        _add(_split_pipe(knn_ans),   weights.get("knn", 0.3)   * max(knn_conf, 0.1))
    if lora_ans:
        _add(_split_pipe(lora_ans),  weights.get("lora", 0.2)  * 0.6)
    if heur_ans:
        _add(_split_pipe(heur_ans),  weights.get("heur", 0.1)  * heur_conf)

    if all(v == 0 for v in option_scores.values()):
        # Total miss — fall back to heuristic only
        return {
            "answer": heur_ans,
            "source": "heuristic_all_fallback",
            "reason": "no source produced a valid Cx",
            "agent": agent_ans, "knn": knn_ans, "lora": lora_ans, "heur": heur_ans,
            "agent_conf": agent_eff, "knn_conf": knn_conf, "heur_conf": heur_conf,
        }

    if multi:
        # Pick top 2 or 3 — bias toward 3 when several options score > 50% of top
        sorted_scores = sorted(option_scores.values(), reverse=True)
        if len(sorted_scores) >= 3 and sorted_scores[2] >= 0.5 * sorted_scores[0]:
            picked = _top_k_from_votes(option_scores, 3)
        else:
            picked = _top_k_from_votes(option_scores, 2)
    else:
        picked = _top_k_from_votes(option_scores, 1)

    return {
        "answer": picked,
        "source": "hybrid_voted",
        "reason": (f"agent_conf={agent_eff:.2f} knn_conf={knn_conf:.2f} "
                   f"weights={weights}"),
        "agent": agent_ans, "knn": knn_ans, "lora": lora_ans, "heur": heur_ans,
        "agent_conf": agent_eff, "knn_conf": knn_conf, "heur_conf": heur_conf,
    }


def tune_weights(scenarios: List[Dict[str, Any]],
                 agent_recs: Dict[str, Dict[str, Any]],
                 lora_recs: Dict[str, Dict[str, Any]],
                 trust_threshold: float = 0.75,
                 min_tools: int = 2,
                 knn_k: int = 10) -> Tuple[Dict[str, float], float, List[Dict[str, Any]]]:
    """Grid search over (W_agent, W_knn, W_lora, W_heur) with sum=1.

    Splits the labelled holdout 75/25 into a tune set + a held-out validation
    set. Weights that win on the tune set get re-evaluated on the val set so
    we can detect grid-search overfitting. Returns (best_weights, val_iou,
    per_grid_log_including_val).
    """
    import random as _random

    labeled = [s for s in scenarios
               if s.get("answer") and s.get("answer") != "To be determined"]
    if not labeled:
        return ({"agent": 0.4, "knn": 0.3, "lora": 0.2, "heur": 0.1}, 0.0, [])

    # Deterministic 75/25 split for reproducibility across reruns.
    rng = _random.Random(42)
    shuffled = list(labeled)
    rng.shuffle(shuffled)
    n_tune = max(1, int(len(shuffled) * 0.75))
    tune_set = shuffled[:n_tune]
    val_set = shuffled[n_tune:]
    print(f"[tune] split: {len(tune_set)} tune / {len(val_set)} val "
          f"(total labelled: {len(labeled)})", file=sys.stderr)

    grids = []
    for w_a in (0.3, 0.4, 0.5, 0.6):
        for w_k in (0.2, 0.3, 0.4):
            for w_l in (0.1, 0.2, 0.3):
                w_h = round(1.0 - w_a - w_k - w_l, 4)
                if w_h < 0.05 or w_h > 0.4:
                    continue
                grids.append({"agent": w_a, "knn": w_k, "lora": w_l, "heur": w_h})

    def _mean_iou(grid_scenarios: List[Dict[str, Any]], w: Dict[str, float]) -> float:
        scores = []
        for s in grid_scenarios:
            sid = s.get("scenario_id", "")
            res = hybrid_fuse_one(s, agent_recs.get(sid, {}), lora_recs.get(sid),
                                  w, trust_threshold, min_tools, knn_k)
            scores.append(iou_score(res["answer"], s.get("answer", "")))
        return sum(scores) / max(len(scores), 1)

    best_tune: Tuple[Optional[Dict[str, float]], float] = (None, -1.0)
    log: List[Dict[str, Any]] = []
    for w in grids:
        tune_iou = _mean_iou(tune_set, w)
        log.append({"weights": w, "tune_iou": tune_iou, "n_tune": len(tune_set)})
        if tune_iou > best_tune[1]:
            best_tune = (w, tune_iou)
    assert best_tune[0] is not None

    # Validate the tune-winner on the held-out set.
    val_iou = _mean_iou(val_set, best_tune[0]) if val_set else best_tune[1]
    log.append({
        "winner": best_tune[0],
        "tune_iou": best_tune[1],
        "val_iou": val_iou,
        "n_val": len(val_set),
    })
    print(f"[tune] winner weights={best_tune[0]}  tune_iou={best_tune[1]:.4f}  "
          f"val_iou={val_iou:.4f}", file=sys.stderr)

    return best_tune[0], val_iou, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_completions", type=Path, required=True)
    ap.add_argument("--test_file", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--score_against_gt", action="store_true",
                    help="If test_file has ground-truth answers, print scores.")
    # ---- Hybrid (4-source) fusion controls. Triggered by env or --hybrid. ----
    ap.add_argument("--hybrid", action="store_true",
                    default=(os.environ.get("ENSEMBLE_HYBRID", "0") == "1"),
                    help="Enable the 4-source confidence-gated fusion path.")
    ap.add_argument("--lora_completions", type=Path, default=None,
                    help="completions.jsonl from the LoRA-attached agentic holdout. "
                         "Required for the hybrid path; ignored otherwise.")
    ap.add_argument("--knn_train", default=os.environ.get(
        "AGENT_FEWSHOT_TRAIN", "data/Phase_1/train.json"))
    ap.add_argument("--knn_cache", default=os.environ.get(
        "AGENT_FEWSHOT_CACHE", "knowledge/processed/train_knn_cache.npz"))
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--trust_threshold", type=float, default=0.75,
                    help="Effective-confidence threshold for trusting the agent's answer raw.")
    ap.add_argument("--min_tools", type=int, default=2,
                    help="Agent must have called this many tools to be trusted.")
    ap.add_argument("--w_agent", type=float, default=0.40)
    ap.add_argument("--w_knn",   type=float, default=0.30)
    ap.add_argument("--w_lora",  type=float, default=0.20)
    ap.add_argument("--w_heur",  type=float, default=0.10)
    ap.add_argument("--tune_weights", type=Path, default=None,
                    help="Path to a LABELED holdout JSON. Runs a grid search over "
                         "(W_agent, W_knn, W_lora, W_heur), picks the highest-IoU "
                         "combination, persists to <out_dir>/weights.json.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    llm_recs = load_llm_completions(args.llm_completions)
    scenarios = json.loads(args.test_file.read_text(encoding="utf-8"))
    print(f"[load] llm_completions={len(llm_recs)}  scenarios={len(scenarios)}")

    # ============================================================
    # HYBRID PATH (4-source confidence-gated fusion)
    # ============================================================
    if args.hybrid:
        # Resolve all paths relative to project root if not already absolute.
        knn_train = args.knn_train if Path(args.knn_train).is_absolute() else (PROJECT_DIR / args.knn_train)
        knn_cache = args.knn_cache if Path(args.knn_cache).is_absolute() else (PROJECT_DIR / args.knn_cache)
        knn_ok = _knn_load_or_build(Path(knn_train), Path(knn_cache))
        if not knn_ok:
            print("[hybrid] WARNING: k-NN not available — its weight will be 0", file=sys.stderr)

        lora_recs: Dict[str, Dict[str, Any]] = {}
        if args.lora_completions:
            lora_recs = load_llm_completions(args.lora_completions)
            print(f"[load] lora_completions={len(lora_recs)}")
        else:
            print("[hybrid] no --lora_completions provided; LoRA weight will be 0")

        # Optional grid-search tuning on labeled holdout
        weights = {"agent": args.w_agent, "knn": args.w_knn,
                   "lora": args.w_lora, "heur": args.w_heur}
        if args.tune_weights and args.tune_weights.exists():
            tune_path = args.tune_weights
            if not tune_path.is_absolute():
                tune_path = PROJECT_DIR / tune_path
            tune_scen = json.loads(tune_path.read_text(encoding="utf-8"))
            # Use the same llm_recs + lora_recs for tuning (assumed to cover holdout).
            print(f"[tune] grid-searching weights on {tune_path} ({len(tune_scen)} scenarios)")
            best_w, best_iou, log = tune_weights(
                tune_scen, llm_recs, lora_recs,
                trust_threshold=args.trust_threshold,
                min_tools=args.min_tools, knn_k=args.knn_k,
            )
            (args.out_dir / "tune_log.json").write_text(
                json.dumps(log, indent=2), encoding="utf-8")
            (args.out_dir / "weights.json").write_text(
                json.dumps({"weights": best_w, "mean_iou": best_iou}, indent=2),
                encoding="utf-8")
            print(f"[tune] best weights = {best_w}  mean_iou={best_iou:.4f}")
            weights = best_w

        # Apply hybrid fusion on the TEST scenarios.
        rows_hybrid: List[Dict[str, str]] = []
        decisions_hybrid: List[Dict[str, Any]] = []
        src_counts: Dict[str, int] = {}
        scores_hybrid: List[float] = []
        gt_by_id = {s["scenario_id"]: s.get("answer", "") for s in scenarios
                    if s.get("answer") and s.get("answer") != "To be determined"}
        for s in scenarios:
            sid = s.get("scenario_id", "")
            res = hybrid_fuse_one(
                s, llm_recs.get(sid, {}), lora_recs.get(sid),
                weights,
                trust_threshold=args.trust_threshold,
                min_tools=args.min_tools, knn_k=args.knn_k,
            )
            rows_hybrid.append({"scenario_id": sid, "answers": res["answer"]})
            decisions_hybrid.append({
                "scenario_id": sid, **{k: v for k, v in res.items() if k != "answer"},
                "answer": res["answer"],
                "is_multi": task_is_multi(s),
            })
            src_counts[res["source"]] = src_counts.get(res["source"], 0) + 1
            if sid in gt_by_id:
                scores_hybrid.append(iou_score(res["answer"], gt_by_id[sid]))

        # Source breakdown
        print()
        print("Hybrid decision breakdown:")
        for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
            pct = 100 * v / max(len(rows_hybrid), 1)
            print(f"  {k:<24s}: {v:>4d}  ({pct:5.1f}%)")
        if scores_hybrid:
            print(f"  HYBRID mean IoU = {sum(scores_hybrid)/len(scores_hybrid):.4f} "
                  f"(n={len(scores_hybrid)})")

        # Write hybrid CSV in Zindi format
        df = pd.DataFrame(rows_hybrid, dtype=str).fillna("")
        df = df.rename(columns={"scenario_id": "ID", "answers": "Track A"})
        df["Track B"] = ""
        df = df[["ID", "Track A", "Track B"]]
        out_csv = args.out_dir / "result_hybrid_zindi.csv"
        df.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}  ({len(df)} rows)")

        (args.out_dir / "decisions_hybrid.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in decisions_hybrid),
            encoding="utf-8")
        # Fall through so the legacy two-tiebreak CSVs are also written below
        # (they're cheap and useful for A/B comparison).

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
