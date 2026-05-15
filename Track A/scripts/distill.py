"""
scripts/distill.py — self-distillation corpus generator (Stage C).

For each scenario in train.json, prompt the running LLM with the GROUND TRUTH
answer and ask it to construct a reasoning trace that arrives there. Keep
only traces whose final \\boxed{...} matches ground truth AND pass a
structural sanity check. Writes traces/train_traces.jsonl.

Resumable: a second run skips scenarios already accepted.

Run after the LLM server is up (default http://localhost:8001/v1):

    python scripts/distill.py \\
        --train_file data/Phase_1/train.json \\
        --output traces/train_traces.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from main import format_scenario_for_prompt, normalize_multi_answer  # noqa: E402
from utils import extract_answer_all  # noqa: E402

API_KEY = os.environ.get("AGENT_API_KEY", "dummy")

TEACHER_PROMPT = """You are producing AGENTIC training data for a smaller reasoning model
that must demonstrate tool-use during the Phase 3 code review of the Telco
Agentic Challenge.

You are given a 5G drive-test scenario, the candidate optimization actions,
and THE CORRECT ANSWER. Produce a clean, agentic reasoning trace that:
  1. Uses ONLY data from the scenario block below. Do not invent timestamps,
     PCIs, cell IDs, or values that are not in the data.
  2. References at least one specific Timestamp from user_plane_data and at
     least one specific PCI from network_configuration_data.
  3. Includes AT LEAST ONE explicit tool-call block emitted as:
         <tool_call>{{"name": "calculate_pathloss", "arguments": {{"time": "...", "pci": ...}}}}</tool_call>
     followed by a short "Tool returned: ...; therefore ..." reflection line.
     Use tool names from this allowlist:
       judge_mainlobe_or_not, calculate_overlap_ratio, calculate_pathloss,
       calculate_horizontal_angle, calculate_tilt_angle, optimize_antenna_gain
     Pick the tool that BEST disambiguates the failure mode you suspect.
     You may invent plausible numeric tool returns (e.g. pathloss=125.3) that
     are CONSISTENT with the scenario data — the smaller model is being
     taught how to reason about tool outputs, not the exact values.
  4. Total length 250-1000 tokens. No padding, no apologies, no caveats.
  5. Follow this diagnostic procedure: scan user-plane for the throughput
     collapse, classify the failure mode (coverage / interference /
     scheduler) by comparing RSRP, SINR, BLER, MCS, RB count, call ≥1 tool
     to verify, then map the mode to the candidate action.
  6. End the trace with a single JSON object on the LAST line:
       {{"answer": "{gt}", "confidence": 0.85, "agree_with_heuristic": true, "tools_used": ["<tool_name(s)>"]}}
     The "answer" MUST equal the correct answer string exactly. Multi-answer
     uses pipe-separated Cx in ascending numeric order (e.g. "C3|C7|C11").

Correct answer: {gt}
Question type: {qtype}

Scenario data:
{scenario_block}

Candidate options:
{options}

Produce the agentic reasoning trace now."""


# ----------------------------- helpers -----------------------------

def build_options_text(options: List[Dict[str, str]]) -> str:
    return "\n".join(f"{o['id']}: {o['label']}" for o in options)


def question_type(task: Dict[str, Any]) -> str:
    desc = (task or {}).get("description", "") or ""
    return "multiple-answer" if "two to four" in desc.lower() or "select two" in desc.lower() else "single-answer"


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
_PCI_RE = re.compile(r"\b\d{2,4}\b")
_CELL_ID_RE = re.compile(r"\b\d{6,}(?:_\d+)?\b")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL)
_JSON_ANS_RE = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)

_ALLOWED_TOOLS = {
    "judge_mainlobe_or_not", "calculate_overlap_ratio", "calculate_pathloss",
    "calculate_horizontal_angle", "calculate_tilt_angle", "optimize_antenna_gain",
}


def structural_check(trace: str, scenario: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, reason). Enforces agentic trace shape:
       - ≥1 well-formed <tool_call>...</tool_call> block using an allowed tool
       - exactly one final answer marker (JSON object or legacy \\boxed{}).

    Set DISTILL_BOXED_ONLY=1 to relax the agentic check: only \\boxed{}
    final answers are required, tool-call blocks become optional. Use this
    fallback when the agentic accept rate is unrecoverably low.
    """
    if not trace or len(trace) < 200:
        return False, "too_short"
    if len(trace) > 8000:
        return False, "too_long"

    boxed_only_mode = os.environ.get("DISTILL_BOXED_ONLY", "0") == "1"

    if not boxed_only_mode:
        # Require at least one tool_call block using an allowed tool.
        tool_blocks = _TOOL_CALL_RE.findall(trace)
        if not tool_blocks:
            return False, "no_tool_call"
        any_allowed = False
        for block in tool_blocks:
            try:
                payload = json.loads(block[block.index("{"):block.rindex("}") + 1])
                if payload.get("name") in _ALLOWED_TOOLS:
                    any_allowed = True
                    break
            except Exception:
                continue
        if not any_allowed:
            return False, "tool_call_invalid"

    # Final-answer marker — JSON preferred, \boxed{} accepted.
    json_objs = _JSON_ANS_RE.findall(trace)
    boxed = _BOXED_RE.findall(trace)
    if not json_objs and len(boxed) != 1:
        return False, "no_final_answer"

    data = scenario.get("data", {}) or {}
    up = data.get("user_plane_data", "") or ""
    cfg = data.get("network_configuration_data", "") or ""

    # Need at least one shared timestamp prefix from user_plane.
    up_timestamps = set(_TS_RE.findall(up))
    if up_timestamps and not any(ts in trace for ts in up_timestamps):
        return False, "no_timestamp"

    # Need at least one PCI mentioned in trace that's in the configuration.
    cfg_pcis = set(m for m in _PCI_RE.findall(cfg) if 0 < int(m) < 1024)
    trace_pcis = set(m for m in _PCI_RE.findall(trace) if 0 < int(m) < 1024)
    if cfg_pcis and not (trace_pcis & cfg_pcis):
        return False, "no_matching_pci"

    # No hallucinated cell IDs (7+ digit underscored IDs in trace must be in scenario).
    cfg_cells = set(_CELL_ID_RE.findall(cfg))
    trace_cells = set(_CELL_ID_RE.findall(trace))
    if trace_cells and not trace_cells.issubset(cfg_cells | {""}):
        return False, "hallucinated_cell"

    return True, "ok"


def load_done(out_path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("scenario_id")
            if sid:
                done[sid] = rec
    return done


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/Phase_1/train.json")
    ap.add_argument("--output", default="traces/train_traces.jsonl")
    ap.add_argument("--model_url", default=os.environ.get("MODEL_URL", "http://localhost:8001/v1"),
                    help="Single model URL. Overridden by --model_urls / LLM_URLS if set.")
    ap.add_argument("--model_urls", default=os.environ.get("LLM_URLS", ""),
                    help="Comma-separated URLs for round-robin failover (e.g. "
                         "http://localhost:8001/v1,http://localhost:8002/v1).")
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap the number of scenarios processed (None = all)")
    ap.add_argument("--attempts_per_scenario", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=1500)
    args = ap.parse_args()

    train_path = Path(args.train_file)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("r", encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = json.load(f)
    if args.max_samples:
        scenarios = scenarios[: args.max_samples]
    print(f"[distill] loaded {len(scenarios)} scenarios from {train_path}")

    done = load_done(out_path)
    print(f"[distill] {len(done)} already accepted in {out_path}; will skip")

    urls = [u.strip() for u in (args.model_urls or args.model_url).split(",") if u.strip()]
    print(f"[distill] llm endpoints: {urls}")
    clients = [OpenAI(base_url=u, api_key=API_KEY, timeout=180.0) for u in urls]
    url_cycle = itertools.cycle(range(len(clients)))
    url_lock = threading.Lock()

    def next_client_idx() -> int:
        with url_lock:
            return next(url_cycle)

    accepted = 0
    rejected = 0
    started = time.time()
    out_f = out_path.open("a", encoding="utf-8")
    summary_path = out_path.parent / "distill_summary.md"

    try:
        for idx, scen in enumerate(scenarios):
            sid = scen.get("scenario_id")
            if not sid or sid in done:
                continue
            gt = scen.get("answer", "") or ""
            task = scen.get("task", {}) or {}
            options = task.get("options", []) or []
            scenario_block = format_scenario_for_prompt(scen)
            prompt = TEACHER_PROMPT.format(
                gt=gt,
                qtype=question_type(task),
                scenario_block=scenario_block,
                options=build_options_text(options),
            )

            this_accepted: Optional[Dict[str, Any]] = None
            this_attempts: List[Dict[str, Any]] = []
            for attempt in range(args.attempts_per_scenario):
                # Round-robin across all configured endpoints. On Connection error,
                # the next attempt will hit a different server automatically.
                ci = next_client_idx()
                try:
                    resp = clients[ci].chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[distill] {sid} attempt {attempt+1} via {urls[ci]} API error: {e}")
                    this_attempts.append({"attempt": attempt + 1, "url": urls[ci], "error": str(e)})
                    continue
                trace = resp.choices[0].message.content or ""
                # Try JSON-format answer first (agentic teacher output), then
                # fall back to legacy \boxed{}.
                pred_raw = ""
                for raw_obj in reversed(_JSON_ANS_RE.findall(trace)):
                    try:
                        pred_raw = str(json.loads(raw_obj).get("answer", "")).strip()
                        if pred_raw:
                            break
                    except Exception:
                        continue
                if not pred_raw:
                    pred_raw = extract_answer_all(trace)
                pred = normalize_multi_answer(pred_raw)
                gt_norm = normalize_multi_answer(gt)
                ok_struct, why = structural_check(trace, scen)
                this_attempts.append({
                    "attempt": attempt + 1,
                    "pred": pred,
                    "ok_struct": ok_struct,
                    "why": why,
                    "len": len(trace),
                })
                if pred == gt_norm and ok_struct:
                    this_accepted = {
                        "scenario_id": sid,
                        "input": scenario_block,
                        "task": task.get("description", ""),
                        "options": options,
                        "trace": trace.strip(),
                        "answer": gt,
                        "qtype": question_type(task),
                        "attempt_used": attempt + 1,
                    }
                    break

            if this_accepted is not None:
                accepted += 1
                out_f.write(json.dumps(this_accepted, ensure_ascii=False) + "\n")
                out_f.flush()
            else:
                rejected += 1

            if (idx + 1) % 20 == 0 or (idx + 1) == len(scenarios):
                elapsed = time.time() - started
                done_so_far = accepted + rejected
                rate = done_so_far / max(elapsed, 1.0)
                eta = (len(scenarios) - len(done) - done_so_far) / max(rate, 1e-6)
                accept_rate = accepted / max(done_so_far, 1)
                print(
                    f"[distill] {idx+1}/{len(scenarios)} "
                    f"accepted={accepted} rejected={rejected} "
                    f"accept_rate={accept_rate:.1%} "
                    f"{rate*60:.1f}/min eta_remaining={eta/60:.0f}min"
                )

            # Early-abort watchdog: if accept rate is unrecoverably low after
            # 200 scenarios, stop and let the orchestrator fall back to
            # \boxed{}-only teacher prompt instead of burning hours.
            # Override via DISTILL_ABORT_MIN_RATE=0 (disable) or a custom value.
            done_so_far = accepted + rejected
            min_rate_env = os.environ.get("DISTILL_ABORT_MIN_RATE", "0.15")
            try:
                abort_min_rate = float(min_rate_env)
            except ValueError:
                abort_min_rate = 0.15
            if abort_min_rate > 0 and done_so_far >= 200:
                live_rate = accepted / done_so_far
                if live_rate < abort_min_rate:
                    print(
                        f"[distill] ABORT: accept_rate={live_rate:.1%} after "
                        f"{done_so_far} scenarios (below {abort_min_rate:.0%}). "
                        "Teacher cannot produce valid agentic traces — "
                        "rerun with DISTILL_BOXED_ONLY=1 to relax the structural check.",
                        file=sys.stderr,
                    )
                    break
    finally:
        out_f.close()
        elapsed = time.time() - started
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"# Distillation summary\n\n")
            f.write(f"- Scenarios processed this run: {accepted + rejected}\n")
            f.write(f"- Already in corpus before run:  {len(done)}\n")
            f.write(f"- Accepted this run:             {accepted}\n")
            f.write(f"- Rejected this run:             {rejected}\n")
            f.write(f"- Total accepted in corpus:      {len(done) + accepted}\n")
            f.write(f"- Wall-clock:                    {elapsed/60:.1f} min\n")
        print(f"[distill] done. accepted={accepted} rejected={rejected} total={len(done)+accepted}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
