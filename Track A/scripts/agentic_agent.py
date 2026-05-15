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

# Heuristic fallback (validated ~0.30 mean IoU on train)
from scripts.build_baseline_submission import pick_answer as heuristic_pick  # noqa: E402
from scripts.build_baseline_submission import is_multi as task_is_multi  # noqa: E402


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

STEP 5 — OPTIONAL TOOL CALL (≤1, only if it disambiguates two actions)
  • judge_mainlobe_or_not(time, pci)
      Disambiguates azimuth-rotation vs tilt-change. If UE is OUTSIDE
      the mainlobe → prefer azimuth. If INSIDE → prefer tilt.
  • calculate_overlap_ratio(pci_serving, pci_neighbor)
      Ratio > 0.3 implicates that neighbor as the interferer.
  • calculate_pathloss(time, pci)
      Confirms coverage degradation when RSRP-based reasoning is ambiguous.
  Do NOT call data-fetch tools (the scenario inlines everything you need).

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
FORMAT — strictly enforced
================================================================
"""

SYSTEM_PROMPT_SINGLE = _SYSTEM_BODY + """\
Single-answer task: the description says "Select the most appropriate".
On the LAST line of your reply, output EXACTLY:
    \\boxed{Cx}
where Cx is ONE option ID. Examples: \\boxed{C7}    \\boxed{C12}    \\boxed{C20}
Do NOT use commas, pipes, multiple options, or any text after \\boxed{}.
Keep reasoning ≤300 tokens before the final line.
"""

SYSTEM_PROMPT_MULTI = _SYSTEM_BODY + """\
Multi-answer task: the description says "Select two to four". Scoring is IoU —
missing a correct option costs as much as adding a wrong one. Pick 2–4 actions;
when in doubt prefer 3 plausible options over 1 confident one.
On the LAST line of your reply, output EXACTLY:
    \\boxed{Cx|Cy|Cz}
with 2–4 option IDs in ASCENDING numeric order, pipe-separated, no spaces.
Examples: \\boxed{C3|C7}    \\boxed{C5|C9|C11|C20}    \\boxed{C2|C8|C16}
Do NOT use commas. Do NOT output a single option for multi-answer tasks.
Keep reasoning ≤300 tokens before the final line.
"""


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

# Tool descriptors lazy-loaded from server.py /tools on first use.
_TOOL_DEFS_CACHE: Optional[List[Dict[str, Any]]] = None


def _fetch_tool_defs(tool_url: str, timeout_s: float = 10.0) -> List[Dict[str, Any]]:
    """Discover the full tool catalog from server.py /tools. Cached for the run."""
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
    if isinstance(raw, list):
        for t in raw:
            if not isinstance(t, dict):
                continue
            fn = t.get("function", t)
            name = fn.get("name") if isinstance(fn, dict) else None
            if not name or name in _EXCLUDE_META:
                continue
            # Normalize to OpenAI tools schema
            if t.get("type") == "function" and "function" in t:
                tools.append(t)
            else:
                tools.append({"type": "function", "function": fn})
    _TOOL_DEFS_CACHE = tools
    print(f"  [tools] loaded {len(tools)} tool descriptors from {tool_url}/tools",
          file=sys.stderr)
    return tools


# --------------------------------------------------------------------- helpers

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_CX_RE = re.compile(r"\bC\d+\b")


def _truncate_scenario(scenario: Dict[str, Any], max_chars: int = 7500) -> str:
    """Render scenario data with priority order. Drop low-value tables if oversize."""
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


def _extract_boxed(text: str, valid_options: List[str]) -> str:
    """Parse \\boxed{Cx} or \\boxed{Cx|Cy|Cz} from model output."""
    if not text:
        return ""
    matches = _BOXED_RE.findall(text)
    if not matches:
        # last-ditch: any Cx mention
        valid = set(valid_options)
        cx = [c for c in _CX_RE.findall(text) if c in valid]
        return cx[0] if cx else ""
    inner = re.sub(r"[{}\s]", "", matches[-1]).lstrip(":").rstrip("./")
    if not inner:
        return ""
    valid = set(valid_options)
    parts = [p for p in inner.split("|") if p in valid]
    if not parts:
        return ""
    parts = sorted(set(parts), key=lambda s: int(re.search(r"\d+", s).group()))
    return "|".join(parts)


def _format_question(scenario: Dict[str, Any]) -> str:
    options = (scenario.get("task") or {}).get("options", []) or []
    options_block = "\n".join(f"  {o['id']}: {o['label']}" for o in options if "id" in o)
    task_desc = (scenario.get("task") or {}).get("description") or ""
    data_block = _truncate_scenario(scenario)
    return (
        f"{data_block}\n\n"
        f"## Task\n{task_desc}\n\n"
        f"## Options\n{options_block}\n\n"
        f"Final answer:"
    )


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


def _agent_turn(
    scenario: Dict[str, Any],
    llm_url: str,
    tool_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
    max_tool_calls: int = 2,
) -> Dict[str, Any]:
    """
    Agentic loop for one scenario. Up to `max_tool_calls` tool calls + 1
    final answer. All server.py tools are available; the LLM picks which.
    Returns dict with text, tool_calls_made, num_tool_calls.
    """
    sid = scenario.get("scenario_id", "")
    is_multi = task_is_multi(scenario)
    system = SYSTEM_PROMPT_MULTI if is_multi else SYSTEM_PROMPT_SINGLE
    question = _format_question(scenario)
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

        # If model produced a boxed answer with no tool calls, we're done
        if not tcs and ("\\boxed{" in text or _BOXED_RE.search(text)):
            return {"text": text, "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}

        if not tcs:
            # No tool, no boxed — break to the explicit final-answer turn
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

    # Final-answer turn (no tools, tighter instruction)
    messages.append({
        "role": "user",
        "content": (
            "Based on the scenario data"
            + (" and tool results" if tool_calls_made else "")
            + ", output ONLY the final answer on the LAST line in this exact format: "
            + ("\\boxed{Cx|Cy|Cz} with 2-4 options in ascending order"
               if is_multi else "\\boxed{Cx}")
            + ". Keep any reasoning under 80 tokens."
        ),
    })
    final = _call_llm(messages, llm_url, model_name, timeout_s, max_tokens, tools=None)
    if final is None:
        return {"text": "", "tool_calls_made": tool_calls_made, "num_tool_calls": len(tool_calls_made)}
    return {"text": final.get("content") or "",
            "tool_calls_made": tool_calls_made,
            "num_tool_calls": len(tool_calls_made)}


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
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--llm_timeout_s", type=float, default=120.0)
    ap.add_argument("--scenario_timeout_s", type=float, default=240.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_tool_calls", type=int, default=1,
                    help="Max number of tool-call turns per scenario before forcing final answer.")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AGENT_WORKERS", "1")),
                    help="Concurrent scenario workers. Set to len(llm_urls) for true GPU parallelism.")
    args = ap.parse_args()

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

        if llm_ok:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    _agent_turn, scenario, url, args.tool_url,
                    args.model_name, args.llm_timeout_s, args.max_tokens,
                    args.max_tool_calls,
                )
                try:
                    res = fut.result(timeout=args.scenario_timeout_s)
                    llm_text = res.get("text", "") or ""
                    tool_calls_made = res.get("tool_calls_made", [])
                    num_tool_calls = res.get("num_tool_calls", 0)
                    answer = _extract_boxed(llm_text, valid_ids)
                except concurrent.futures.TimeoutError:
                    print(f"  [timeout] scenario {sid[:8]} (worker={worker_idx})",
                          file=sys.stderr)
                    fut.cancel()

        source = "llm"
        if not answer:
            answer = heuristic_pick(scenario)
            source = "heuristic"

        elapsed = time.time() - t0
        rec = {
            "scenario_id": sid,
            "answer": answer,
            "source": source,
            "worker": worker_idx,
            "llm_url": url,
            "num_tool_calls": num_tool_calls,
            "tool_calls": tool_calls_made,
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
