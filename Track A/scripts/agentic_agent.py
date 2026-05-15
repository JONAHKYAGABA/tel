"""
scripts/agentic_agent.py — agentic LLM agent for Track A.

Pipeline per scenario:
  1. Build a tight prompt: truncated scenario data + options + tool descriptors
  2. Ask the LLM with tools=[judge_mainlobe_or_not, calculate_overlap_ratio,
     calculate_pathloss]. Temperature 0.
  3. If the LLM responds with tool_calls, execute them against server.py
     (port 7860 by default), feed results back as a second turn.
  4. Extract \\boxed{Cx} (single) or \\boxed{Cx|Cy|Cz} (multi) from the final
     response. If empty / malformed, fall back to the deterministic heuristic.
  5. Per-scenario hard timeout 60s.

Outputs:
  <out_dir>/completions.jsonl     # resumable per-scenario log
  <out_dir>/result.csv            # scenario_id, answers
  <out_dir>/result_v1_raw.csv     # same
  <out_dir>/result_v2_multi_recall.csv
  <out_dir>/result_v3_insurance.csv

Auto-scores against ground truth if scenarios are labeled.

Run:
    python scripts/agentic_agent.py \\
        --test_file data/Phase_1/test.json \\
        --out_dir   eval/results/agentic

Assumes llm_server is up at $LLM_URL (default http://localhost:8001) and
server.py is up at $TOOL_URL (default http://localhost:7860).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

# Heuristic fallback (validated ~0.30 mean IoU on train) + diagnosis helper
from scripts.build_baseline_submission import pick_answer as heuristic_pick  # noqa: E402
from scripts.build_baseline_submission import is_multi as task_is_multi  # noqa: E402
from scripts.build_baseline_submission import heuristic_diagnosis  # noqa: E402
from scripts.build_baseline_submission import (  # noqa: E402
    parse_table, find_drop, classify_mode,
)


# --------------------------------------------------------------------- RAG

_RAG_MODEL = None
_RAG_CHUNKS: Optional[List[Dict[str, Any]]] = None
_RAG_EMBS = None  # numpy array (n_chunks, dim)

# k-NN few-shot retrieval over the training set (separate index from RAG).
# Same MiniLM model is reused via _ensure_sbert() below.
_KNN_FEATS = None       # numpy array (n_train, dim)
_KNN_META: List[Dict[str, Any]] = []   # parallel list of dicts


def _init_rag(kb_dir: Path) -> bool:
    """Lazy-load sentence-transformers + KB chunks + embeddings.
    Returns True if RAG is usable. Idempotent."""
    global _RAG_MODEL, _RAG_CHUNKS, _RAG_EMBS
    if _RAG_MODEL is not None:
        return True
    chunks_path = kb_dir / "chunks.json"
    embs_path = kb_dir / "embeddings.npy"
    if not chunks_path.exists() or not embs_path.exists():
        print(f"[rag] index not found at {kb_dir} — run scripts/build_kb_index.py",
              file=sys.stderr)
        return False
    if _ensure_sbert() is None:
        return False
    _RAG_CHUNKS = json.loads(chunks_path.read_text(encoding="utf-8"))
    import numpy as np
    _RAG_EMBS = np.load(embs_path)
    print(f"[rag] indexed {len(_RAG_CHUNKS)} chunks (dim={_RAG_EMBS.shape[1]})",
          file=sys.stderr)
    return True


def _rag_query(scenario: Dict[str, Any]) -> str:
    """Short query derived from the scenario's task + options (NOT the full data,
    which would dilute the signal)."""
    task_desc = ((scenario.get("task") or {}).get("description") or "")[:300]
    opts = " | ".join(
        (o.get("label") or "")
        for o in ((scenario.get("task") or {}).get("options") or [])
    )[:700]
    return f"5G RAN troubleshooting. {task_desc} Candidate actions: {opts}"


def _retrieve_context(scenario: Dict[str, Any], k: int = 3,
                      max_chars_per_chunk: int = 700) -> str:
    """Top-k cosine search over the KB. Returns a markdown block or '' if disabled."""
    import numpy as np
    if _RAG_MODEL is None or _RAG_CHUNKS is None or _RAG_EMBS is None:
        return ""
    q = _RAG_MODEL.encode([_rag_query(scenario)], convert_to_numpy=True)[0].astype("float32")
    q_norm = np.linalg.norm(q) + 1e-8
    e_norms = np.linalg.norm(_RAG_EMBS, axis=1) + 1e-8
    sims = (_RAG_EMBS @ q) / (e_norms * q_norm)
    top_idx = np.argsort(-sims)[:k]
    parts = []
    for i in top_idx:
        c = _RAG_CHUNKS[int(i)]
        text = c["text"][:max_chars_per_chunk]
        src = c.get("source", "kb")
        parts.append(f"From {src}:\n{text}")
    return ("## Reference Knowledge (top-{} retrieved 5G/3GPP passages)\n\n"
            .format(k)) + "\n\n---\n\n".join(parts)


# ------------------------------------------------------- k-NN few-shot retrieval

def _ensure_sbert():
    """Load (and cache) the shared MiniLM sentence-transformer used by both
    RAG and k-NN. Returns the model, or None if sentence-transformers is not
    available. Calling this multiple times is a no-op after the first load."""
    global _RAG_MODEL
    if _RAG_MODEL is not None:
        return _RAG_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[knn] sentence-transformers not installed", file=sys.stderr)
        return None
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    model_id = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"[knn/rag] loading shared embedding model {model_id}", file=sys.stderr)
    _RAG_MODEL = SentenceTransformer(model_id)
    return _RAG_MODEL


def _scenario_signal_stats(scenario: Dict[str, Any]) -> Dict[str, float]:
    """Compact signal stats for a scenario: RSRP/SINR mean+std, throughput drop %.
    Falls back to neutral values when parsing fails."""
    up = (scenario.get("data") or {}).get("user_plane_data", "") or ""
    df = parse_table(up)
    if df.empty:
        return {"rsrp_mean": -100.0, "rsrp_std": 0.0,
                "sinr_mean": 0.0, "sinr_std": 0.0, "tput_drop_pct": 0.0}
    rsrp_c = next((c for c in df.columns if "RSRP" in c and "Serving" in c), None)
    sinr_c = next((c for c in df.columns if "SINR" in c and "Serving" in c), None)
    tput_c = next((c for c in df.columns if "DL Throughput" in c), None)
    import pandas as pd
    def _stats(col: Optional[str]):
        if col is None:
            return (0.0, 0.0)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return (0.0, 0.0)
        return (float(s.mean()), float(s.std() or 0.0))
    rsrp_mean, rsrp_std = _stats(rsrp_c) if rsrp_c else (-100.0, 0.0)
    sinr_mean, sinr_std = _stats(sinr_c) if sinr_c else (0.0, 0.0)
    drop_pct = 0.0
    if tput_c is not None:
        tp = pd.to_numeric(df[tput_c], errors="coerce").dropna()
        if not tp.empty and tp.max() > 0:
            drop_pct = float((tp.max() - tp.min()) / tp.max() * 100.0)
    return {"rsrp_mean": rsrp_mean, "rsrp_std": rsrp_std,
            "sinr_mean": sinr_mean, "sinr_std": sinr_std,
            "tput_drop_pct": drop_pct}


def _knn_feature_text(scenario: Dict[str, Any]) -> str:
    """Build the embedding text for k-NN. Compact, discriminative features:
      - heuristic-classified failure mode (semantic anchor)
      - signal stats (numeric profile)
      - task FIRST SENTENCE only (long task descriptions add noise)
      - option labels truncated to ≤60 chars each (preserve action verbs,
        drop cell-specific suffixes that hurt cross-scenario matching)
      - serving PCI + cell count (topology proxy)
    """
    stats = _scenario_signal_stats(scenario)
    diag = find_drop(parse_table((scenario.get("data") or {}).get("user_plane_data", "") or ""))
    mode = classify_mode(diag)
    # Task: take everything up to the first period (or 200 chars).
    task_raw = ((scenario.get("task") or {}).get("description") or "")
    if "." in task_raw[:300]:
        task_short = task_raw.split(".", 1)[0][:200]
    else:
        task_short = task_raw[:200]
    # Options: truncate per-label to keep action verbs but drop noisy cell IDs.
    opts = (scenario.get("task") or {}).get("options") or []
    opt_text = " | ".join((o.get("label") or "")[:60] for o in opts)[:600]
    # Topology proxy
    cfg = (scenario.get("data") or {}).get("network_configuration_data", "") or ""
    cfg_df = parse_table(cfg)
    n_cells = len(cfg_df) if not cfg_df.empty else 0
    serving_pci = diag["serving_pci"] if diag else "?"
    return (
        f"failure_mode={mode}\n"
        f"serving_pci={serving_pci} n_cells={n_cells}\n"
        f"rsrp={stats['rsrp_mean']:.0f}dBm sinr={stats['sinr_mean']:.0f}dB "
        f"drop={stats['tput_drop_pct']:.0f}%\n"
        f"task={task_short}\n"
        f"options={opt_text}"
    )


def _knn_signal_summary(scenario: Dict[str, Any]) -> str:
    s = _scenario_signal_stats(scenario)
    return f"RSRP≈{s['rsrp_mean']:.0f}dBm, SINR≈{s['sinr_mean']:.0f}dB, drop≈{s['tput_drop_pct']:.0f}%"


def _knn_init(train_path: str, cache_path: str) -> bool:
    """Build (or load) the k-NN embedding cache for training scenarios.
    Idempotent: skips work if `_KNN_FEATS` is already populated."""
    global _KNN_FEATS, _KNN_META
    if _KNN_FEATS is not None and _KNN_META:
        return True
    import numpy as np
    cache = Path(cache_path)
    if cache.exists():
        try:
            data = np.load(cache, allow_pickle=False)
            _KNN_FEATS = data["feats"]
            _KNN_META = json.loads(str(data["meta"]))
            print(f"[knn] loaded cache from {cache} ({len(_KNN_META)} scenarios, "
                  f"dim={_KNN_FEATS.shape[1]})", file=sys.stderr)
            return True
        except Exception as exc:
            print(f"[knn] cache load failed ({exc}); rebuilding", file=sys.stderr)

    sbert = _ensure_sbert()
    if sbert is None:
        return False

    train = Path(train_path)
    if not train.exists():
        # try resolve relative to project root
        train = PROJECT_DIR / train_path
    if not train.exists():
        print(f"[knn] train file not found at {train_path}", file=sys.stderr)
        return False

    with train.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    print(f"[knn] embedding {len(scenarios)} training scenarios "
          f"(this is a one-time cost, ~5 min on CPU)", file=sys.stderr)

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for s in scenarios:
        ans = s.get("answer") or ""
        if not ans or ans == "To be determined":
            continue
        texts.append(_knn_feature_text(s))
        meta.append({
            "scenario_id": s.get("scenario_id", ""),
            "answer": ans,
            "signal_summary": _knn_signal_summary(s),
            "task_short": ((s.get("task") or {}).get("description") or "")[:200],
        })

    feats = sbert.encode(texts, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True).astype("float32")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=feats, meta=json.dumps(meta, ensure_ascii=False))
    _KNN_FEATS = feats
    _KNN_META = meta
    print(f"[knn] cache built and saved to {cache} "
          f"({len(meta)} labelled scenarios, dim={feats.shape[1]})", file=sys.stderr)
    return True


def _knn_retrieve(scenario: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    """Top-k cosine search over the training scenarios. Returns a list of
    {scenario_id, answer, signal_summary, task_short, sim}."""
    if _KNN_FEATS is None or not _KNN_META:
        return []
    sbert = _ensure_sbert()
    if sbert is None:
        return []
    import numpy as np
    q = sbert.encode([_knn_feature_text(scenario)], convert_to_numpy=True)[0].astype("float32")
    q_norm = np.linalg.norm(q) + 1e-8
    f_norms = np.linalg.norm(_KNN_FEATS, axis=1) + 1e-8
    sims = (_KNN_FEATS @ q) / (f_norms * q_norm)
    sid = scenario.get("scenario_id", "")
    # Exclude self-match if the test scenario happens to be in the train set
    top_idx = np.argsort(-sims)
    out: List[Dict[str, Any]] = []
    for i in top_idx:
        m = _KNN_META[int(i)]
        if m.get("scenario_id") == sid:
            continue
        out.append({**m, "sim": float(sims[int(i)])})
        if len(out) >= k:
            break
    return out


def _format_fewshot_block(similar: List[Dict[str, Any]]) -> str:
    if not similar:
        return ""
    lines = ["## Similar Solved Cases (training-set retrieval — for pattern reference, not copy-paste)"]
    for i, ex in enumerate(similar, 1):
        lines.append(f"### Example {i} (sim={ex['sim']:.2f})")
        lines.append(f"- Signals: {ex.get('signal_summary', '')}")
        task = ex.get("task_short", "")
        if task:
            lines.append(f"- Task: {task}...")
        lines.append(f"- Correct answer: {ex.get('answer', '')}")
    return "\n".join(lines)


def _required_min_tools(scenario: Dict[str, Any], floor: int = 0) -> int:
    """Adaptive minimum tool calls per scenario.

    Multi-answer scenarios and scenarios with many neighbors are genuinely
    harder and benefit from chained tool evidence. Simple single-answer
    scenarios with few neighbors can be answered with one tool (or zero, if
    the heuristic is already high-confidence) without losing accuracy.

    Returns max(floor, derived). `floor` is the global --min_tools setting.
    """
    is_multi_task = task_is_multi(scenario)
    cfg = (scenario.get("data") or {}).get("network_configuration_data", "") or ""
    df = parse_table(cfg)
    n_cells = len(df) if not df.empty else 0
    # n_cells is the number of cell rows in network_configuration_data; a
    # cheap proxy for "how many candidate sources/neighbors does this case have".
    if is_multi_task or n_cells > 5:
        derived = 2
    else:
        derived = 1
    return max(int(floor), derived)


def _tool_penalty(num_tools: int, required: int) -> float:
    """Smoothed confidence multiplier given how many tools were actually called."""
    if num_tools >= required:
        return 1.0
    if num_tools >= 1:
        return 0.75
    return 0.5


def _format_heuristic_block(diag: Dict[str, Any]) -> str:
    """Render the heuristic diagnosis as the agent's opening context. The
    LLM is explicitly told this is a tool output it may override."""
    return (
        "## Heuristic Tool Diagnosis (auto-generated — verify with tools, override if wrong)\n"
        f"- Failure mode: {diag.get('failure_mode', 'UNKNOWN')}\n"
        f"- RSRP drop: {diag.get('delta_rsrp_db', 0.0):.1f} dB  |  "
        f"SINR drop: {diag.get('delta_sinr_db', 0.0):.1f} dB\n"
        f"- Suggested answer: {diag.get('recommended_answer', '')}\n"
        f"- Heuristic confidence: {diag.get('confidence', 0.0):.2f}\n"
        f"- Rationale: {diag.get('reasoning', '')}"
    )


# --------------------------------------------------------------------- prompts

_SYSTEM_BODY = """\
You are a senior 5G RAN drive-test troubleshooting engineer. You will see ONE \
scenario with: network_configuration_data (gNodeB/Cell IDs, PCI, azimuth, \
tilt, height, ARFCN, TxPower, A2/A3/A5 thresholds), user_plane_data \
(per-timestamp PCI, RSRP, SINR, DL throughput, BLER, MCS, RB count, top-N \
neighbors), signaling_plane_data (NREventA2/A3/A5, NRRandomAccess, \
NRRRCReestablishAttempt), traffic_data (PRB utilization, weak-coverage ratio, \
CCE Allocation Success Rate), mr_data (serving + neighbor PCI/RSRP).

================================================================
DIAGNOSTIC PROCEDURE — follow EVERY step, do not skip
================================================================

STEP 1 — LOCATE THE THROUGHPUT COLLAPSE
  Scan user_plane_data for the row where DL throughput drops ≥50% vs the
  trailing 3-row mean. Record:
    t_drop          = timestamp of the collapse
    PCI_serving     = serving PCI at t_drop
    gNB_cell_serv   = the gNodeB_Cell pair whose PCI matches PCI_serving
                      (look it up in network_configuration_data)

STEP 2 — CLASSIFY FAILURE MODE
  Compute the deltas between the 3 rows BEFORE t_drop and the row AT t_drop:
    ΔRSRP   = RSRP_pre − RSRP_at      (positive = degradation)
    ΔSINR   = SINR_pre − SINR_at      (positive = degradation)
    ΔBLER   = BLER_at − BLER_pre      (positive = degradation)
    ΔRB     = RB_pre − RB_at          (positive = scheduler starvation)

  Decision tree:
    A. COVERAGE (serving cell):   ΔRSRP > 6 dB AND ΔSINR > 5 dB
    B. INTERFERENCE (neighbor):   ΔSINR > 5 dB AND ΔRSRP < 3 dB AND a neighbor
                                  in mr_data has RSRP within 3 dB of serving
    C. QUALITY:                   ΔBLER > 20 pp AND low MCS at t_drop
                                  → treat the same as INTERFERENCE
    D. SCHEDULER / PDCCH:         ΔRSRP < 3 dB AND ΔSINR < 3 dB AND MCS healthy
                                  BUT ΔRB > 50% OR low CCE Allocation Success
                                  in traffic_data
    E. MOBILITY (handover):       NRRRCReestablishAttempt fires near t_drop OR
                                  repeated NREventA3 firings on the same neighbor

STEP 3 — CROSS-CHECK signaling_plane_data in [t_drop − 5s, t_drop + 5s]
  • NRRRCReestablishAttempt        → missing neighbor relation OR A3/A5 wrong
  • Repeated NREventA3 same target → ping-pong, A3 Offset wrong on serving cell
  • RSRP low but no NREventA2      → A2 threshold too strict; lower
                                     CovInterFreqA2RsrpThld on the serving cell

STEP 4 — CROSS-CHECK traffic_data
  • High Downlink Weak Coverage Ratio          → confirms COVERAGE
  • Low Downlink CCE Allocation Success Rate   → confirms PDCCH / SCHEDULER
  • High PRB Utilization + throughput dip      → load/scheduling

STEP 5 — OPTIONAL TOOL CALL (≤1, only if it disambiguates two actions).
  You have these 6 computation tools available (already loaded — call by name):

    • judge_mainlobe_or_not(time, pci)
        Returns whether the UE is inside the mainlobe of serving cell `pci`
        at `time`. OUTSIDE → prefer azimuth action; INSIDE → prefer tilt.

    • calculate_overlap_ratio(pci_serving, pci_neighbor)
        Returns coverage overlap (0–1). > 0.3 implicates that neighbor as
        the interferer; prefer tilt-down / decrease-power on the neighbor.

    • calculate_pathloss(time, pci)
        Returns the pathloss in dB at `time` for cell `pci`. Confirms
        coverage degradation when RSRP-based reasoning is ambiguous.

    • calculate_horizontal_angle(time, pci)
        Returns the bearing (degrees) from cell `pci` to UE at `time`.
        Use to choose how much azimuth to rotate.

    • calculate_tilt_angle(time, pci)
        Returns the recommended tilt for cell `pci` at `time`. Use to
        choose between "lift the tilt" vs "press down the tilt".

    • optimize_antenna_gain(time, pci)
        Returns a suggested antenna gain optimisation for cell `pci`.

  Skip tools entirely if the inline data is unambiguous. Do NOT call any
  data-fetch tool (e.g. get_user_plane_data) — that data is already inlined
  in this prompt.

STEP 6 — MAP FAILURE MODE → ACTION TEMPLATE → CORRECT CELL
  The 22 options are templated actions parameterised by a SPECIFIC gNodeB_Cell.
  Match the failure mode to the template AND verify the cell in the option
  matches the diagnosed cell:

    COVERAGE on serving:
      → "Increase transmission power for <serving_cell>"
      → "Lift the tilt of <serving_cell> by N degrees"
      → "Adjust the azimuth of <serving_cell> by N degrees"
      → "Decrease CovInterFreqA2RsrpThld for <serving_cell>"

    INTERFERENCE from neighbor:
      → "Press down the tilt of <neighbor_cell> by N degrees"
      → "Adjust the azimuth of <neighbor_cell> by N degrees"
      → "Decrease transmission power for <neighbor_cell>"
      → "Increase A3 Offset threshold for <serving_cell>"
      → "Add neighbor relationship between <serving> and <neighbor>"

    PDCCH / SCHEDULER:
      → "Modify PdcchOccupiedSymbolNum to 2SYM for <cell>"
      → "Check test server and transmission issues"

    AMBIGUOUS / no clear pattern → "Insufficient data; more data is needed
    for judgment" ONLY if RSRP, SINR, BLER, MCS, RB are all healthy at t_drop.

================================================================
WORKED EXAMPLES (for calibration — do NOT copy verbatim)
================================================================

Example 1 (COVERAGE on serving):
  user_plane shows RSRP dropping from −85 dBm to −103 dBm and SINR from
  12 dB to 2 dB at t_drop on PCI 451. PCI 451 is gNodeB_Cell 3279943_1
  in network_configuration_data. mr_data shows no strong competing neighbor.
  → COVERAGE on 3279943_1. The matching options are "Increase transmission
  power for 3279943_1" or "Lift the tilt of 3279943_1 by 4 degrees".
  Pick the one whose cell ID matches.

Example 2 (INTERFERENCE from neighbor):
  RSRP stays ≈ −90 dBm but SINR collapses from 14 dB to 1 dB at t_drop on
  PCI 451 (3279943_1). mr_data shows neighbor PCI 488 (3267220_2) at
  RSRP −89 dBm.
  → INTERFERENCE from 3267220_2. Candidates: "Press down the tilt of
  3267220_2 by 4 degrees", "Adjust the azimuth of 3267220_2 by 24 degrees",
  "Increase A3 Offset threshold for 3279943_1".

Example 3 (PDCCH / SCHEDULER):
  RSRP −88 dBm, SINR 15 dB, BLER 2%, MCS 24 — all healthy. But RB count
  drops from 100 to 4, and traffic_data shows low CCE Allocation Success
  Rate for 3279943_1.
  → PDCCH on 3279943_1. Action: "Modify PdcchOccupiedSymbolNum to 2SYM
  for 3279943_1".

================================================================
HEURISTIC + FEW-SHOT CONTEXT — read this BEFORE the scenario data
================================================================

The user message you receive is structured like this:

  [Heuristic Tool Diagnosis]     ← auto-generated diagnosis (mode + suggested answer)
  [Similar Solved Cases]          ← 3 training scenarios with known correct answers
  [Reference Knowledge]           ← retrieved 3GPP / 5G doc passages
  [Network/Userplane/...]         ← the actual scenario data tables
  [Task + Options]                ← the question and the option list

Treat the heuristic diagnosis as a tool result. If it agrees with your own
read of the data AND its confidence is ≥0.70, agree with it. Override it
only when you can name a specific contradiction in the data. The similar
cases show what kinds of answers are correct for similar signal profiles.

================================================================
TOOL-CALLING REQUIREMENT (agentic — required for code review)
================================================================

You MUST call AT LEAST 2 tools from the allowlist before emitting your
final answer, EXCEPT when the heuristic's mode is AMBIG / NO_DROP (in
which case the answer is almost always "Insufficient data" and tool
calls add no information).

Recommended pairings (call BOTH; the second confirms the first):
  • COVERAGE suspected → calculate_pathloss + judge_mainlobe_or_not
  • INTERFERENCE suspected → calculate_overlap_ratio + calculate_horizontal_angle
  • SCHEDULER / PDCCH suspected → optimize_antenna_gain + calculate_tilt_angle
  • MOBILITY / handover suspected → calculate_overlap_ratio + judge_mainlobe_or_not

The tool-call protocol is the existing <tool_call>{...}</tool_call> block
emitted by the harness — emit one tool call at a time and wait for the
"tool" role response before the next.

Skipping tool calls is treated as low confidence and will be downweighted
by the post-processing fusion layer.

================================================================
FORMAT — strictly enforced
================================================================
"""

_FINAL_OUTPUT_BLOCK = """\

Preferred final format (on the LAST line of your final-turn reply):

    {"answer": "C7", "confidence": 0.85, "agree_with_heuristic": true, \
"tools_used": ["calculate_pathloss", "judge_mainlobe_or_not"]}

For multi-answer tasks: "answer" is pipe-separated Cx in ASCENDING order,
e.g. "C3|C7" or "C5|C9|C11|C20". `confidence` is a self-estimate in [0,1];
use ≥0.75 only when tools confirmed the heuristic diagnosis.

Legacy fallback (still accepted by the parser): \\boxed{Cx} for single,
\\boxed{Cx|Cy|Cz} for multi. Use ONLY if the JSON format is impossible.
Keep reasoning ≤300 tokens before the final line.
"""

SYSTEM_PROMPT_SINGLE = _SYSTEM_BODY + """\
Single-answer task: the description says "Select the most appropriate".
Output ONE option ID in the JSON `answer` field (e.g. "C7").
""" + _FINAL_OUTPUT_BLOCK

SYSTEM_PROMPT_MULTI = _SYSTEM_BODY + """\
Multi-answer task: the description says "Select two to four". Scoring is IoU —
missing a correct option costs as much as adding a wrong one. Pick 2–4 actions;
when in doubt prefer 3 plausible options over 1 confident one.
Output 2-4 IDs pipe-separated in ASCENDING numeric order in the JSON `answer`
field (e.g. "C3|C7|C11"). Do NOT use commas. Do NOT emit a single option for
multi-answer tasks.
""" + _FINAL_OUTPUT_BLOCK


# Function name -> server.py URL path. Full set of tools exposed by server.py;
# matches Environment.endpoint_mapper in main.py. Meta endpoints (health,
# get_all_scenario, get_available_tools) are NOT exposed to the LLM as tools.
ENDPOINT_MAP = {
    "get_config_data":                "/config-data",
    "get_user_plane_data":            "/user-plane-data",
    "get_throughput_logs":            "/throughput-logs",
    "get_cell_info":                  "/cell-info",
    "get_gnodeb_location":            "/gnodeb-location",
    "get_user_location":              "/user-location",
    "get_serving_cell_pci":           "/serving-cell-pci",
    "get_serving_cell_rsrp":          "/serving-cell-rsrp",
    "get_serving_cell_sinr":          "/serving-cell-sinr",
    "get_rbs_allocated_to_user":      "/rbs-allocated-to-user",
    "get_neighboring_cells_pci":      "/neighboring-cells-pci",
    "get_neighboring_cell_rsrp":      "/neighboring-cell-rsrp",
    "get_signaling_plane_event_log":  "/signaling-plane-event-log",
    "get_all_cells_pci":              "/all-cells-pci",
    "get_kpi_data":                   "/get_kpi_data",
    "get_mr_data":                    "/get_mr_data",
    "judge_mainlobe_or_not":          "/judge_mainlobe",
    "calculate_horizontal_angle":     "/calculate_horizontal_angle",
    "calculate_tilt_angle":           "/calculate_tilt_angle",
    "calculate_pathloss":             "/calculate_pathloss",
    "calculate_overlap_ratio":        "/calculate_overlap_ratio",
    "optimize_antenna_gain":          "/optimize_antenna_gain",
}

_EXCLUDE_META = {"health", "get_all_scenario", "get_available_tools"}

# Only expose to the LLM the tools that produce DERIVED information not already
# in the inlined scenario data. Data-fetch tools (get_serving_cell_rsrp, etc.)
# retrieve slices of data the agent already has — they just bloat the prompt
# without changing what the model can know. The 6 here are the actual compute
# tools that change what the model knows. Override via AGENT_TOOLS_ALLOWLIST.
_DEFAULT_ALLOW_TOOLS = {
    "judge_mainlobe_or_not",
    "calculate_overlap_ratio",
    "calculate_pathloss",
    "calculate_horizontal_angle",
    "calculate_tilt_angle",
    "optimize_antenna_gain",
}
_env_allow = os.environ.get("AGENT_TOOLS_ALLOWLIST", "").strip()
if _env_allow:
    _ALLOW_TOOLS = {t.strip() for t in _env_allow.split(",") if t.strip()}
elif _env_allow == "ALL":
    _ALLOW_TOOLS = None  # no filter
else:
    _ALLOW_TOOLS = _DEFAULT_ALLOW_TOOLS

# Tool descriptors lazy-loaded from server.py /tools on first use.
_TOOL_DEFS_CACHE: Optional[List[Dict[str, Any]]] = None


def _fetch_tool_defs(tool_url: str, timeout_s: float = 10.0) -> List[Dict[str, Any]]:
    """Discover the tool catalog from server.py /tools, filtered to compute
    tools only (by default). Cached for the run."""
    global _TOOL_DEFS_CACHE
    if _TOOL_DEFS_CACHE is not None:
        return _TOOL_DEFS_CACHE
    headers: Dict[str, str] = {}
    bearer = os.environ.get("TOOL_BEARER_TOKEN") or os.environ.get("AGENT_API_KEY")
    if bearer and not bearer.startswith("sk-dummy"):
        # Phase 2 cloud sandbox uses X-API-Token; keep Authorization for local fallback
        headers["X-API-Token"] = bearer
        headers["Authorization"] = f"Bearer {bearer}"
    verify = os.environ.get("TOOL_VERIFY_TLS", "1") == "1"
    try:
        r = requests.get(f"{tool_url.rstrip('/')}/tools", timeout=timeout_s,
                         headers=headers, verify=verify)
        r.raise_for_status()
        raw = r.json()
    except Exception as exc:
        print(f"  [tools] failed to fetch /tools: {exc} — using empty catalog", file=sys.stderr)
        _TOOL_DEFS_CACHE = []
        return _TOOL_DEFS_CACHE

    tools: List[Dict[str, Any]] = []
    n_total = n_kept = 0
    if isinstance(raw, list):
        for t in raw:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", t)
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name or name in _EXCLUDE_META:
                continue
            n_total += 1
            if _ALLOW_TOOLS is not None and name not in _ALLOW_TOOLS:
                continue
            n_kept += 1
            # Normalize to OpenAI tools schema
            if t.get("type") == "function" and "function" in t:
                tools.append(t)
            else:
                tools.append({"type": "function", "function": fn})
    _TOOL_DEFS_CACHE = tools
    allow_str = "ALL" if _ALLOW_TOOLS is None else ",".join(sorted(_ALLOW_TOOLS))
    print(f"  [tools] kept {n_kept}/{n_total} from {tool_url}/tools "
          f"(allowlist={allow_str})", file=sys.stderr)
    return tools


# --------------------------------------------------------------------- helpers

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_CX_RE = re.compile(r"\bC\d+\b")


def _truncate_scenario(scenario: Dict[str, Any], max_chars: Optional[int] = None) -> str:
    """Render scenario data with priority order. Drop low-value tables if oversize.

    Pass `max_chars` explicitly or set AGENT_SCENARIO_MAX_CHARS env var.
    Default 5000 chars (~1250 tokens). On a Turing GPU with the 35B model,
    the self-attention prefill score matrix scales as O(seq_len²), so cutting
    1000 input tokens drops peak attention memory by ~360 MB per layer.
    Lower this (e.g. 3500) if you see CUDA OOMs at inference time.
    """
    if max_chars is None:
        try:
            max_chars = int(os.environ.get("AGENT_SCENARIO_MAX_CHARS", "5000"))
        except ValueError:
            max_chars = 5000
    d = scenario.get("data") or {}
    parts: List[str] = []

    def add(label: str, key: str) -> None:
        val = d.get(key)
        if val:
            parts.append(f"## {label}\n```\n{val.strip()}\n```")

    add("Network Configuration", "network_configuration_data")
    add("User-Plane Time Series", "user_plane_data")
    add("Signaling Plane Events", "signaling_plane_data")
    add("Cell-Level Traffic KPIs", "traffic_data")
    add("Measurement Reports", "mr_data")

    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text
    # Drop mr_data first (least diagnostic), then traffic_data.
    for drop_section in ("Measurement Reports", "Cell-Level Traffic KPIs"):
        text = re.sub(
            rf"## {drop_section}\n```.*?```\n*", "", text, flags=re.DOTALL
        )
        if len(text) <= max_chars:
            return text
    # Last resort: hard truncate user_plane_data's middle
    return text[:max_chars]


def _normalize_cx_answer(raw: str, valid_options: List[str]) -> str:
    """Validate + sort an Cx-pipe answer string. Returns '' if no valid Cx
    remain after filtering against the option allowlist."""
    if not raw:
        return ""
    valid = set(valid_options)
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    parts = [p for p in parts if p in valid]
    if not parts:
        return ""
    parts = sorted(set(parts), key=lambda s: int(re.search(r"\d+", s).group()))
    return "|".join(parts)


# Strict shape for a final answer string: C7 or C3|C7|C12 (any length).
_CX_ANSWER_RE = re.compile(r"^C\d+(\|C\d+)*$")
# Capture the LAST JSON object in the text (greedy from end). Used to extract
# the model's final {"answer": ...} record.
_LAST_JSON_RE = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)


def _extract_answer(text: str, valid_options: List[str]) -> Dict[str, Any]:
    """Layered parser: try JSON → \\boxed{} → bare Cx mention.

    Returns a dict {answer, confidence, agree_with_heuristic, tools_used,
                    source_format}. 'answer' is '' if nothing parseable.
    """
    out: Dict[str, Any] = {
        "answer": "",
        "confidence": 0.0,
        "agree_with_heuristic": None,
        "tools_used": [],
        "source_format": "none",
    }
    if not text:
        return out

    # 1) JSON path — find the LAST occurrence so any earlier exemplars in the
    # prompt or chain-of-thought scratch don't shadow the real answer.
    json_matches = _LAST_JSON_RE.findall(text)
    if json_matches:
        for raw_obj in reversed(json_matches):
            try:
                obj = json.loads(raw_obj)
            except json.JSONDecodeError:
                # Tolerate a trailing comma or unquoted keys: try a lenient retry
                fixed = re.sub(r",\s*}", "}", raw_obj)
                try:
                    obj = json.loads(fixed)
                except Exception:
                    continue
            ans_raw = str(obj.get("answer", "")).strip()
            ans = _normalize_cx_answer(ans_raw, valid_options)
            if not ans:
                continue
            try:
                conf = float(obj.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            tools_used = obj.get("tools_used") or []
            if not isinstance(tools_used, list):
                tools_used = []
            agree = obj.get("agree_with_heuristic")
            if isinstance(agree, str):
                agree = agree.lower() == "true"
            return {
                "answer": ans,
                "confidence": max(0.0, min(1.0, conf)),
                "agree_with_heuristic": agree if isinstance(agree, bool) else None,
                "tools_used": [str(t) for t in tools_used][:8],
                "source_format": "json",
            }

    # 2) \boxed{} path — preserve legacy behavior.
    matches = _BOXED_RE.findall(text)
    if matches:
        inner = re.sub(r"[{}\s]", "", matches[-1]).lstrip(":").rstrip("./")
        ans = _normalize_cx_answer(inner, valid_options)
        if ans:
            return {
                "answer": ans, "confidence": 0.5,
                "agree_with_heuristic": None, "tools_used": [],
                "source_format": "boxed",
            }

    # 3) Bare Cx fallback.
    valid = set(valid_options)
    cx = [c for c in _CX_RE.findall(text) if c in valid]
    if cx:
        return {
            "answer": cx[0], "confidence": 0.3,
            "agree_with_heuristic": None, "tools_used": [],
            "source_format": "bare_cx",
        }
    return out


def _extract_boxed(text: str, valid_options: List[str]) -> str:
    """Backward-compatible shim. Returns just the answer string."""
    return _extract_answer(text, valid_options)["answer"]


def _format_question(scenario: Dict[str, Any], rag_block: str = "",
                     heur_block: str = "", fewshot_block: str = "") -> str:
    options = (scenario.get("task") or {}).get("options", []) or []
    options_block = "\n".join(f"  {o['id']}: {o['label']}" for o in options if "id" in o)
    task_desc = (scenario.get("task") or {}).get("description") or ""
    data_block = _truncate_scenario(scenario)
    parts = []
    # Order: heuristic (decision-relevant tool output) → fewshot (calibration
    # examples) → rag (3GPP background) → scenario data → task → options.
    # This puts the most concrete signals nearest the LLM's recency window.
    if heur_block:
        parts.append(heur_block)
    if fewshot_block:
        parts.append(fewshot_block)
    if rag_block:
        parts.append(rag_block)
    parts.append(data_block)
    parts.append(f"## Task\n{task_desc}")
    parts.append(f"## Options\n{options_block}")
    parts.append("Final answer (JSON preferred, see system prompt):")
    return "\n\n".join(parts)


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    scenario_id: str,
    tool_url: str,
    timeout_s: float = 10.0,
) -> str:
    """Call server.py tool endpoint (local or Phase-2 cloud).
    Phase-2 cloud uses X-API-Token header (NOT Authorization: Bearer)."""
    endpoint = ENDPOINT_MAP.get(name)
    if endpoint is None:
        endpoint = "/" + name.replace("_", "-")
    headers: Dict[str, str] = {}
    if scenario_id:
        headers["X-Scenario-Id"] = scenario_id
    bearer = os.environ.get("TOOL_BEARER_TOKEN") or os.environ.get("AGENT_API_KEY")
    if bearer and not bearer.startswith("sk-dummy"):
        # Phase 2 cloud sandbox auth: X-API-Token header
        headers["X-API-Token"] = bearer
        # Also send Authorization for any vanilla server.py instances
        headers["Authorization"] = f"Bearer {bearer}"
    # Cloud endpoints serve over HTTPS with self-signed certs sometimes; tolerate that
    verify = os.environ.get("TOOL_VERIFY_TLS", "1") == "1"
    try:
        r = requests.get(
            f"{tool_url.rstrip('/')}{endpoint}",
            params=arguments,
            headers=headers,
            timeout=timeout_s,
            verify=verify,
        )
        if r.status_code != 200:
            return json.dumps({"error": f"status {r.status_code}", "detail": r.text[:200]})
        text = r.text or ""
        if len(text) > 1500:
            text = text[:1500] + " ...[truncated]"
        return text
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _call_llm(
    messages: List[Dict[str, Any]],
    llm_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Single chat completion call. Returns assistant message dict or None.
    Pass tools=[] (or None) to disable tool-use for this turn."""
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        r = requests.post(
            f"{llm_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {os.environ.get('AGENT_API_KEY', 'sk-dummy')}"},
        )
        if r.status_code != 200:
            print(f"  [llm-{r.status_code}] {r.text[:200]}", file=sys.stderr)
            return None
        return r.json().get("choices", [{}])[0].get("message") or {}
    except Exception as exc:
        print(f"  [llm-error] {exc}", file=sys.stderr)
        return None


def _looks_final(text: str) -> bool:
    """Cheap test: does the assistant text contain a final-answer marker?
    Either a JSON object with an 'answer' key or a \\boxed{} block."""
    if not text:
        return False
    if "\\boxed{" in text or _BOXED_RE.search(text):
        return True
    if _LAST_JSON_RE.search(text):
        return True
    return False


def _agent_turn(
    scenario: Dict[str, Any],
    llm_url: str,
    tool_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
    max_tool_calls: int = 2,
    rag_k: int = 0,
    use_fewshot: bool = False,
    fewshot_k: int = 3,
    use_heur_collab: bool = False,
) -> Dict[str, Any]:
    """
    Agentic loop for one scenario. Up to `max_tool_calls` tool calls + 1
    final answer. Tools are filtered to compute tools by default.

    Optional context blocks (default off, gated by env / CLI flags):
      • use_heur_collab → prepend the heuristic diagnosis as opening context
      • use_fewshot     → prepend top-k similar training scenarios w/ answers
      • rag_k > 0       → prepend top-k 3GPP doc passages
    Returns dict with text, tool_calls_made, num_tool_calls.
    """
    sid = scenario.get("scenario_id", "")
    is_multi = task_is_multi(scenario)
    system = SYSTEM_PROMPT_MULTI if is_multi else SYSTEM_PROMPT_SINGLE

    heur_block = ""
    if use_heur_collab:
        try:
            heur_block = _format_heuristic_block(heuristic_diagnosis(scenario))
        except Exception as exc:
            print(f"  [heur-block] {sid[:8]} failed: {exc}", file=sys.stderr)

    fewshot_block = ""
    if use_fewshot and _KNN_FEATS is not None:
        try:
            sim = _knn_retrieve(scenario, k=fewshot_k)
            fewshot_block = _format_fewshot_block(sim)
        except Exception as exc:
            print(f"  [knn-block] {sid[:8]} failed: {exc}", file=sys.stderr)

    rag_block = _retrieve_context(scenario, k=rag_k) if rag_k > 0 else ""
    question = _format_question(scenario, rag_block=rag_block,
                                heur_block=heur_block, fewshot_block=fewshot_block)
    tool_defs = _fetch_tool_defs(tool_url)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    tool_calls_made: List[Dict[str, Any]] = []

    for turn in range(max_tool_calls):
        msg = _call_llm(messages, llm_url, model_name, timeout_s, max_tokens, tools=tool_defs)
        if msg is None:
            return {"text": "", "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}

        tcs = msg.get("tool_calls") or []
        text = msg.get("content") or ""

        # If model produced a final answer (JSON or \boxed{}) with no tool calls, we're done.
        if not tcs and _looks_final(text):
            return {"text": text, "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}

        if not tcs:
            # No tool, no final answer — break to the explicit final-answer turn
            messages.append({"role": "assistant", "content": text})
            break

        # Execute the FIRST tool call (one per turn)
        tc = tcs[0]
        fn = (tc.get("function") or {}).get("name", "")
        args_str = (tc.get("function") or {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except json.JSONDecodeError:
            args = {}
        result_str = _execute_tool(fn, args, sid, tool_url)
        tool_calls_made.append({
            "name": fn, "arguments": args, "result_head": result_str[:200],
        })
        messages.append({"role": "assistant", "content": text, "tool_calls": [tc]})
        messages.append({
            "role": "tool",
            "tool_call_id": tc.get("id", f"call_{turn}"),
            "content": result_str,
        })

    # Final-answer turn (no tools, tighter instruction).
    # Preferred JSON format; legacy \boxed{} still accepted by the parser.
    tools_used_names = sorted({tc.get("name", "") for tc in tool_calls_made if tc.get("name")})
    tools_used_hint = (
        f' (you called: {", ".join(tools_used_names)})' if tools_used_names else ""
    )
    if is_multi:
        ans_example = '{"answer": "C3|C7|C11", "confidence": 0.82, ' \
                      '"agree_with_heuristic": true, "tools_used": ' \
                      f'{json.dumps(tools_used_names)}}}'
    else:
        ans_example = '{"answer": "C7", "confidence": 0.85, ' \
                      '"agree_with_heuristic": true, "tools_used": ' \
                      f'{json.dumps(tools_used_names)}}}'
    messages.append({
        "role": "user",
        "content": (
            "Based on the scenario data"
            + (" and tool results" if tool_calls_made else "")
            + tools_used_hint
            + ", output your final answer on the LAST line as a single JSON object, "
              "e.g.:\n    " + ans_example
            + ("\n(multi-answer task: 2-4 options in ASCENDING order in `answer`, "
               "pipe-separated)" if is_multi else "")
            + "\nKeep any reasoning before that under 80 tokens. "
              "Legacy \\boxed{...} is still accepted if you cannot emit JSON."
        ),
    })
    final = _call_llm(messages, llm_url, model_name, timeout_s, max_tokens, tools=None)
    if final is None:
        return {"text": "", "tool_calls_made": tool_calls_made,
                "num_tool_calls": len(tool_calls_made),
                "heuristic_suggestion": heur_block and heuristic_diagnosis(scenario) or None}
    return {"text": final.get("content") or "",
            "tool_calls_made": tool_calls_made,
            "num_tool_calls": len(tool_calls_made),
            "heuristic_suggestion": (heuristic_diagnosis(scenario) if use_heur_collab else None)}


# --------------------------------------------------------------------- runner

def _load_completions(path: Path) -> Dict[str, str]:
    done: Dict[str, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                if sid:
                    done[sid] = rec.get("answer", "")
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llm_url", default=os.environ.get("LLM_URL", "http://localhost:8001"))
    ap.add_argument("--llm_urls", default=os.environ.get("LLM_URLS"),
                    help="Comma-separated list of LLM URLs for parallel inference "
                         "(one per GPU). Overrides --llm_url. e.g. "
                         "'http://localhost:8001,http://localhost:8002'")
    ap.add_argument("--tool_url", default=os.environ.get("TOOL_URL", "http://localhost:7860"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--llm_timeout_s", type=float, default=300.0)
    ap.add_argument("--scenario_timeout_s", type=float, default=480.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_tool_calls", type=int, default=1,
                    help="Max number of tool-call turns per scenario before forcing final answer.")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AGENT_WORKERS", "1")),
                    help="Concurrent scenario workers. Set to len(llm_urls) for true GPU parallelism.")
    ap.add_argument("--use_rag", action="store_true",
                    help="Retrieve top-k chunks from knowledge/processed and prepend to prompt.")
    ap.add_argument("--rag_k", type=int, default=int(os.environ.get("RAG_K", "3")),
                    help="Number of KB chunks to retrieve per scenario when --use_rag.")
    ap.add_argument("--rag_dir", default=os.environ.get("RAG_DIR", "knowledge/processed"),
                    help="Directory with chunks.json + embeddings.npy.")
    # ---- New: k-NN few-shot + heuristic collaboration toggles ----
    ap.add_argument("--use_fewshot", action="store_true",
                    default=(os.environ.get("AGENT_USE_FEWSHOT", "0") == "1"),
                    help="Inject top-k similar training scenarios as in-prompt few-shot examples.")
    ap.add_argument("--fewshot_k", type=int,
                    default=int(os.environ.get("AGENT_FEWSHOT_K", "3")))
    ap.add_argument("--fewshot_train",
                    default=os.environ.get("AGENT_FEWSHOT_TRAIN", "data/Phase_1/train.json"))
    ap.add_argument("--fewshot_cache",
                    default=os.environ.get("AGENT_FEWSHOT_CACHE",
                                           "knowledge/processed/train_knn_cache.npz"))
    ap.add_argument("--use_heur_collab", action="store_true",
                    default=(os.environ.get("AGENT_USE_HEUR_COLLAB", "0") == "1"),
                    help="Show the heuristic diagnosis to the LLM as opening context.")
    ap.add_argument("--min_tools", type=int,
                    default=int(os.environ.get("AGENT_MIN_TOOLS", "0")),
                    help="Confidence penalty if LLM emitted fewer than N tool calls. "
                         "Set to 2 in agentic-improved holdout for Phase 3 code review.")
    args = ap.parse_args()

    # Initialise RAG if requested
    rag_active_k = 0
    if args.use_rag:
        kb_path = (PROJECT_DIR / args.rag_dir).resolve()
        if _init_rag(kb_path):
            rag_active_k = args.rag_k
            print(f"[rag] active: top-{rag_active_k} retrieval from {kb_path}")
        else:
            print(f"[rag] disabled — falling back to no-RAG mode", file=sys.stderr)

    # Initialise k-NN few-shot cache if requested
    fewshot_active = False
    if args.use_fewshot:
        train_path = (PROJECT_DIR / args.fewshot_train).resolve()
        cache_path = (PROJECT_DIR / args.fewshot_cache).resolve()
        if _knn_init(str(train_path), str(cache_path)):
            fewshot_active = True
            print(f"[knn] active: top-{args.fewshot_k} few-shot retrieval "
                  f"({len(_KNN_META)} training scenarios indexed)")
        else:
            print("[knn] disabled — falling back to no-few-shot mode", file=sys.stderr)

    if args.use_heur_collab:
        print("[heur-collab] active: heuristic diagnosis will be shown in every prompt")
    if args.min_tools > 0:
        print(f"[min-tools] enforcing: confidence will be halved when num_tool_calls < {args.min_tools}")

    # Resolve URL list: --llm_urls (comma-sep) overrides --llm_url
    if args.llm_urls:
        llm_urls = [u.strip() for u in args.llm_urls.split(",") if u.strip()]
    else:
        llm_urls = [args.llm_url]
    if args.workers < 1:
        args.workers = 1
    # Cap workers to number of URLs (one worker per LLM instance)
    args.workers = min(args.workers, len(llm_urls))
    print(f"[run] llm_urls = {llm_urls}  workers = {args.workers}")

    test_path = (PROJECT_DIR / args.test_file).resolve()
    out_dir = (PROJECT_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = out_dir / "completions.jsonl"
    csv_path = out_dir / "result.csv"

    if not test_path.exists():
        print(f"FATAL: {test_path} not found", file=sys.stderr)
        return 1

    # Health checks for ALL llm_urls (one per GPU)
    healthy_urls: List[str] = []
    for url in llm_urls:
        try:
            h = requests.get(f"{url}/health", timeout=5).json()
            if h.get("status") == "ok":
                healthy_urls.append(url)
                print(f"[llm] {url} -> {h}")
            else:
                print(f"[llm] {url} not ok: {h}", file=sys.stderr)
        except Exception as exc:
            print(f"[llm] {url} not reachable: {exc}", file=sys.stderr)
    llm_ok = len(healthy_urls) > 0
    if not llm_ok:
        print("[run] NO LLMs reachable — all answers will be heuristic fallback", file=sys.stderr)
        healthy_urls = llm_urls  # placeholder so loop has something
    # Effective worker count = min(requested, healthy LLMs)
    eff_workers = min(args.workers, max(len(healthy_urls), 1)) if llm_ok else 1
    print(f"[run] healthy LLMs: {len(healthy_urls)} | effective workers: {eff_workers}")

    tool_ok = False
    try:
        r = requests.get(f"{args.tool_url}/health", timeout=5, verify=False)
        tool_ok = r.status_code in (200, 401, 403)
        print(f"[tool] {args.tool_url} -> {r.status_code}")
    except Exception as exc:
        # Try with verify=False if not already
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = requests.get(f"{args.tool_url}/health", timeout=5, verify=False)
            tool_ok = r.status_code in (200, 401, 403)
            print(f"[tool] {args.tool_url} -> {r.status_code} (no-verify)")
        except Exception as exc2:
            print(f"[tool] not reachable: {exc2}", file=sys.stderr)
    if not tool_ok:
        print("[run] tool server unreachable — agent will run without tool execution", file=sys.stderr)

    with test_path.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    done = _load_completions(completions_path)
    rows: List[Dict[str, str]] = [
        {"scenario_id": sid, "answers": ans} for sid, ans in done.items()
    ]
    pending = [s for s in scenarios if s.get("scenario_id") not in done]
    print(f"[run] {len(scenarios)} scenarios, {len(done)} already done, "
          f"{len(pending)} remaining")

    # Counters + locks for parallel workers
    import threading
    counters_lock = threading.Lock()
    write_lock = threading.Lock()
    counters = {"llm": 0, "fb": 0, "tool_used": 0, "done": 0}
    f_jsonl = completions_path.open("a", encoding="utf-8")
    t_start = time.time()

    def _process_one(scenario: Dict[str, Any], worker_idx: int) -> None:
        """Process ONE scenario on the LLM URL assigned to this worker."""
        sid = scenario.get("scenario_id", "")
        options = (scenario.get("task") or {}).get("options", []) or []
        valid_ids = [o["id"] for o in options if "id" in o]
        url = healthy_urls[worker_idx % len(healthy_urls)]

        t0 = time.time()
        llm_text = ""
        tool_calls_made: List[Dict[str, Any]] = []
        num_tool_calls = 0
        answer = ""
        parsed: Dict[str, Any] = {}
        heur_suggest: Optional[Dict[str, Any]] = None

        if llm_ok:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    _agent_turn, scenario, url, args.tool_url,
                    args.model_name, args.llm_timeout_s, args.max_tokens,
                    args.max_tool_calls, rag_active_k,
                    fewshot_active, args.fewshot_k, args.use_heur_collab,
                )
                try:
                    res = fut.result(timeout=args.scenario_timeout_s)
                    llm_text = res.get("text", "") or ""
                    tool_calls_made = res.get("tool_calls_made", [])
                    num_tool_calls = res.get("num_tool_calls", 0)
                    heur_suggest = res.get("heuristic_suggestion")
                    parsed = _extract_answer(llm_text, valid_ids)
                    answer = parsed.get("answer", "")
                except concurrent.futures.TimeoutError:
                    print(f"  [timeout] scenario {sid[:8]} (worker={worker_idx})",
                          file=sys.stderr)
                    fut.cancel()

        source = "llm"
        source_format = parsed.get("source_format", "none") if parsed else "none"
        if not answer:
            answer = heuristic_pick(scenario)
            source = "heuristic"
            source_format = "heuristic"

        # Effective confidence: downweight if the agent didn't call enough tools.
        # The required count is ADAPTIVE — multi-answer / many-neighbor scenarios
        # demand 2 tools, simple single-answer scenarios demand 1.
        # args.min_tools acts as a global FLOOR (e.g. set to 2 to never relax
        # below 2 calls). Used downstream by submit_ensemble.py's fusion gate.
        conf = float(parsed.get("confidence", 0.0)) if parsed else 0.0
        required_tools = _required_min_tools(scenario, floor=args.min_tools)
        effective_conf = conf * _tool_penalty(num_tool_calls, required_tools)

        elapsed = time.time() - t0
        rec = {
            "scenario_id": sid,
            "answer": answer,
            "source": source,
            "source_format": source_format,
            "worker": worker_idx,
            "llm_url": url,
            "num_tool_calls": num_tool_calls,
            "required_min_tools": required_tools,
            "tool_calls": tool_calls_made,
            "tools_used": parsed.get("tools_used", []),
            "confidence": conf,
            "effective_confidence": effective_conf,
            "agree_with_heuristic": parsed.get("agree_with_heuristic"),
            "heuristic_suggestion": heur_suggest,
            "elapsed_s": round(elapsed, 2),
            "llm_text_head": (llm_text or "")[:200],
        }

        with write_lock:
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            rows.append({"scenario_id": sid, "answers": answer})
        with counters_lock:
            if source == "llm":
                counters["llm"] += 1
            else:
                counters["fb"] += 1
            if num_tool_calls > 0:
                counters["tool_used"] += 1
            counters["done"] += 1
            d = counters["done"]
            running = time.time() - t_start
            avg = running / max(d, 1)
            eta = avg * (len(pending) - d)
            print(
                f"[{d:4d}/{len(pending)}] {sid[:8]} ans={answer:<18s} "
                f"src={source:9s} tool={num_tool_calls} {elapsed:5.1f}s "
                f"w{worker_idx} url=...{url[-5:]}  "
                f"(llm={counters['llm']} fb={counters['fb']} "
                f"tool_used={counters['tool_used']}, eta={eta/60:.1f}min)",
                flush=True,
            )
            # Flush CSV every 10 completions
            if d % 10 == 0 or d == len(pending):
                _write_csv(rows, csv_path)

    # Dispatch pending scenarios round-robin to workers (true GPU parallelism)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers) as pool:
            futures = []
            for i, sc in enumerate(pending):
                futures.append(pool.submit(_process_one, sc, i % eff_workers))
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"[worker-error] {exc}", file=sys.stderr)
    finally:
        f_jsonl.close()
        _write_csv(rows, csv_path)

    # Pull counters back into local names for summary block
    n_llm = counters["llm"]
    n_fb = counters["fb"]
    n_with_tool = counters["tool_used"]

    # Three Zindi-ready variants (identical for now; later we can diverge multi-recall)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "result_v1_raw.csv", index=False)
    df.to_csv(out_dir / "result_v2_multi_recall.csv", index=False)
    df.to_csv(out_dir / "result_v3_insurance.csv", index=False)

    # Auto-score if labeled
    labeled = [s for s in scenarios if s.get("answer") and s.get("answer") != "To be determined"]
    if labeled:
        ans_by_id = {r["scenario_id"]: r["answers"] for r in rows}
        cx_re = re.compile(r"^C\d+(\|C\d+)*$")
        n_scored = 0
        score_total = 0.0
        n_multi_correct = n_multi_total = n_single_correct = n_single_total = 0
        n_empty = n_malformed = 0
        for s in labeled:
            sid = s.get("scenario_id")
            pred = ans_by_id.get(sid, "")
            gt = s.get("answer", "")
            if not pred: n_empty += 1
            elif not cx_re.match(pred): n_malformed += 1
            if "|" in gt:
                n_multi_total += 1
                p, g = set(pred.split("|")) if pred else set(), set(gt.split("|"))
                iou = len(p & g) / max(len(p | g), 1)
                score_total += iou
                if iou == 1.0: n_multi_correct += 1
            else:
                n_single_total += 1
                ok = pred == gt
                score_total += 1.0 if ok else 0.0
                if ok: n_single_correct += 1
            n_scored += 1
        print()
        print(f"=== AGENTIC HOLDOUT SCORE ===")
        print(f"  scored {n_scored}")
        print(f"  mean   : {score_total/max(n_scored,1):.4f}")
        print(f"  single : {n_single_correct}/{n_single_total} exact "
              f"({100*n_single_correct/max(n_single_total,1):.1f}%)")
        print(f"  multi  : {n_multi_correct}/{n_multi_total} full IoU=1.0 "
              f"({100*n_multi_correct/max(n_multi_total,1):.1f}%)")
        print(f"  empty/malformed: {n_empty}/{n_malformed}")
        print(f"  tool used      : {n_with_tool}")
        print(f"  heuristic fb   : {n_fb}")

    print()
    print(f"=== DONE ===")
    print(f"  total       : {len(rows)}")
    print(f"  llm answers : {n_llm}")
    print(f"  heuristic fb: {n_fb}")
    print(f"  with tool   : {n_with_tool}")
    print(f"  csv         : {csv_path}")
    return 0


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    sys.exit(main())
