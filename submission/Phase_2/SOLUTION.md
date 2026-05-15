# Track A — Telco Troubleshooting Agentic Challenge: Solution Document

**Team:** Kyagaba Jonah
**Track:** A — Wireless RAN troubleshooting
**Phase:** 2 (May 2026)
**Repo:** https://github.com/JONAHKYAGABA/tel
**Base model:** `Qwen/Qwen3.5-35B-A3B` (35 B-param MoE, ~3 B active per token; constrained by competition rules)

---

## 1. Executive Summary

We built a hybrid agent that combines:

- A **local LLM server** (Qwen3.5-35B-A3B, 4-bit bitsandbytes) serving OpenAI-compatible chat completions
- A **deterministic heuristic** (RSRP/SINR/BLER decision tree) as both a safety net and a hard fallback
- A **RAG retrieval layer** over 13 526 chunks of 3GPP TS specs (38.331 RRC, 38.300 NR, 38.214/215 PHY, 38.133 measurements) and 10 open-access arXiv papers
- An **agentic loop** that calls 6 compute tools (`judge_mainlobe_or_not`, `calculate_pathloss`, `calculate_overlap_ratio`, `calculate_horizontal_angle`, `calculate_tilt_angle`, `optimize_antenna_gain`) on the Phase 2 cloud sandbox
- A **LoRA self-distillation pipeline** that uses the base model as a teacher (given the ground-truth answer) to construct CoT traces, then fine-tunes a LoRA adapter on those traces
- A single orchestrator script (`scripts/run_all_today.sh`) that runs everything end-to-end and produces three Zindi-format submission CSVs

The whole pipeline is **resumable** at every phase — each step skips if its output already exists, so a crash mid-run does not waste prior compute.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        OUR MACHINE (marconi, 2× RTX 8000)          │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  llm_server (port 8001)                                     │  │
│   │  Primary:   vLLM 0.21 with tensor parallel (TP=1 or TP=2)   │  │
│   │             + continuous batching + PagedAttention          │  │
│   │  Fallback:  transformers + bitsandbytes 4-bit               │  │
│   │  Quant:     NF4 + double quant, fp16 compute                │  │
│   │  Optional:  LoRA adapter (peft) attached at startup         │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│                              │ OpenAI /v1/chat/completions          │
│                              │                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  agentic_agent.py  (the orchestrator-side worker)           │  │
│   │  • Parallel ThreadPoolExecutor (1 worker per LLM URL)       │  │
│   │  • Per-scenario: retrieve top-k RAG chunks, build prompt,   │  │
│   │    call LLM with tools, execute any tool_calls, get final   │  │
│   │    \\boxed{Cx} answer; fall back to heuristic on failure    │  │
│   │  • Thread-safe writes to completions.jsonl + result.csv     │  │
│   │  • Resumable via completions.jsonl                          │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                  │                              │                   │
│                  │ cosine search                │ HTTPS + X-API-Token│
│                  ▼                              ▼                   │
│   ┌──────────────────────────────┐  ┌─────────────────────────┐   │
│   │  RAG knowledge base (LOCAL)  │  │                         │   │
│   │  knowledge/raw/              │  │                         │   │
│   │  ├── 8 × 3GPP TS PDFs        │  │                         │   │
│   │  ├── 10 × arXiv PDFs         │  │                         │   │
│   │  ├── ShareTechNote HTML      │  │                         │   │
│   │  └── Wikipedia primers       │  │                         │   │
│   │  knowledge/processed/        │  │                         │   │
│   │  ├── chunks.json  (13 526)   │  │                         │   │
│   │  └── embeddings.npy          │  │                         │   │
│   │      (MiniLM-L6-v2, 384-d)   │  │                         │   │
│   └──────────────────────────────┘  └──────────┬──────────────┘   │
└────────────────────────────────────────────────│───────────────────┘
                                                 │
                                                 ▼
                       ┌──────────────────────────────────────────┐
                       │  ZINDI CLOUD SANDBOX (Phase 2)           │
                       │  https://124.71.227.61/no   (HK/overseas)│
                       │                                          │
                       │  Auth: X-API-Token: <team-token>         │
                       │  Per-scenario context: X-Scenario-Id     │
                       │                                          │
                       │  22 endpoints; we expose 6 compute tools │
                       │  to the LLM (the others duplicate inline │
                       │  scenario data — no point spending tokens│
                       │  on them).                               │
                       └──────────────────────────────────────────┘
```

---

## 3. Component-by-Component

### 3.1 LLM Serving — `scripts/llm_server.py` + `scripts/llm_server_vllm.py`

We provide two interchangeable OpenAI-compatible LLM servers:

| | `llm_server.py` (transformers + bnb) | `llm_server_vllm.py` (vLLM) |
|---|---|---|
| Backbone | HF transformers + bnb 4-bit NF4 | vLLM 0.21.0 |
| Batching | One request at a time | Continuous batching |
| GPU usage | Pipeline parallel via `device_map="auto"`, with explicit `max_memory` per GPU to leave activation headroom | Tensor parallel (`--tensor-parallel-size N`) |
| LoRA | `peft.PeftModel.from_pretrained(base, lora_path)` | `--enable-lora --lora-modules adapter=<path>` |
| Throughput | ~30–60 s / scenario | ~10–20 s / scenario (when both GPUs are usable) |
| Stability on Turing (RTX 8000) | Robust | Flaky for Qwen3.5 MoE + bnb + TP=2 (vLLM 0.21 bug, see §9) |

`run_all_today.sh` tries vLLM first via `start_llm_parallel`; if vLLM fails to start, it falls back to the transformers+bnb server. The user can force a path via `USE_VLLM=0`.

Key env knobs:

```
USE_VLLM=1                    # 0 forces transformers+bnb
QUANT_MODE=bitsandbytes       # or awq, gptq, none
TENSOR_PARALLEL_SIZE=1 or 2   # vLLM only; TP=1 sidesteps a MoE+bnb shape bug
GPU_MEMORY_UTILIZATION=0.85   # vLLM only
MAX_MODEL_LEN=8192            # vLLM only
LLM_PER_GPU_GIB=42            # transformers path; leaves ~5 GB / GPU for activations
```

### 3.2 Agentic Loop — `scripts/agentic_agent.py`

Per scenario:

1. **Retrieve RAG context** (if `--use_rag`) — top-k cosine search over `knowledge/processed/embeddings.npy`. Query is the task description + option labels (NOT the bulk data, which would dilute retrieval).
2. **Build the prompt** — system prompt (~1 800 tokens of structured 7-step diagnostic procedure with 3 worked examples) + optional RAG block + truncated scenario data + options + "Final answer:".
3. **First LLM call** with `tools=[6 compute tools]`. If the model emits a `tool_calls` block, execute the first one against the Phase 2 cloud sandbox (with `X-API-Token` + `X-Scenario-Id` headers).
4. **Tool-result turn** — feed the tool result back as a `tool` message and call the LLM again, this time without tools, asking for the final `\\boxed{Cx}` line.
5. **Extract** `\\boxed{Cx}` (or `\\boxed{Cx|Cy|Cz}` for multi-answer). On parse failure or LLM timeout, **fall back** to the heuristic.

Per-scenario hard cap: 240 s wall-clock (`scenario_timeout_s`). Per-LLM-call timeout: 75 s. Up to 1 tool call by default (`--max_tool_calls`). Configurable max-tokens (default 768).

Parallelism: `ThreadPoolExecutor(max_workers=N)` where N matches the number of healthy LLM URLs. Each worker is pinned to one URL and processes scenarios round-robin.

Thread-safe writes: `write_lock` around `completions.jsonl` and `result.csv` flushes; `counters_lock` around the live ETA/`(llm=X fb=Y tool_used=Z)` print.

Heuristic fallback (~26.8 % validated IoU on training data) lives in `scripts/build_baseline_submission.py` — pure-Python decision tree based on the system prompt's classification rules, used both as in-process fallback and as a standalone `--no_llm` baseline submission.

### 3.3 RAG Knowledge Base — `scripts/download_rag_docs.sh` + `scripts/build_kb_index.py`

**Sources** (all CC-BY / open access, no auth):

- **3GPP TS specs via ETSI** (8 PDFs covering the exact KPIs in the scenarios):
  - TS 38.331 (RRC), 38.300 (NR overall), 38.214/215 (PHY/measurements), 38.213 (PDCCH scheduling), 38.211 (PHY channels), 38.321 (MAC), 38.133 (measurement requirements)
- **arXiv** (10 PDFs): 5G survey, drive-test ML, handover failure analysis, PDCCH, beam management, coverage optimization
- **ShareTechNote** (best-effort HTML pulls of 5G_PDCCH, 5G_PDSCH, 5G_MIMO, 5G_PowerControl)
- **Wikipedia** primers (5G NR, RSRP, Handover, PCI)

**Indexing** (`build_kb_index.py`):

- Parses `.txt`, `.md`, `.html` (via trafilatura with regex fallback), and `.pdf` (via pypdf)
- Splits paragraphs > 100 chars, hard-splits anything > 1500 chars into 1200-char windows
- Embeds with `sentence-transformers/all-MiniLM-L6-v2` (384-d, ~80 MB, CPU-only, ~30 s for 13 k chunks)
- Outputs `chunks.json` + `embeddings.npy`

**Retrieval** (in `agentic_agent.py`):

```python
q = _RAG_MODEL.encode([_rag_query(scenario)])[0]
sims = (_RAG_EMBS @ q) / (||_RAG_EMBS|| × ||q||)
top_k = argsort(-sims)[:k]
```

Top-k chunks are prepended to the LLM prompt as a "## Reference Knowledge" markdown block.

### 3.4 Tool Calling

The Phase 2 sandbox publishes 22 endpoints via `GET /tools`. Of these, we expose **only 6 to the LLM**:

| Tool | Returns | When the LLM should call |
|---|---|---|
| `judge_mainlobe_or_not(time, pci)` | bool | Distinguish azimuth-rotation vs tilt-change |
| `calculate_overlap_ratio(pci_serving, pci_neighbor)` | float ∈ [0,1] | Confirm a neighbor is the interferer (> 0.3) |
| `calculate_pathloss(time, pci)` | dB | Confirm coverage degradation when RSRP is ambiguous |
| `calculate_horizontal_angle(time, pci)` | degrees | Decide how much azimuth to rotate |
| `calculate_tilt_angle(time, pci)` | degrees | Decide lift vs press-down |
| `optimize_antenna_gain(time, pci)` | suggestion | Final-step gain tuning |

The other 16 endpoints are data-fetchers (`get_serving_cell_rsrp`, `get_user_plane_data`, etc.) — the data they return is **already inlined** in the scenario JSON, so we omit them to save ~1 500 prompt tokens per call. Filter is in `_ALLOW_TOOLS`; override via `AGENT_TOOLS_ALLOWLIST` env.

**Authentication** (discovered empirically via direct testing): the sandbox returns `401 Missing API Token. Please provide 'X-API-Token' header` despite the README saying "Authorization: Bearer". We send both headers on every tool call:

```
X-API-Token: <team-token>
Authorization: Bearer <team-token>
X-Scenario-Id: <uuid>
```

TLS: the sandbox uses a self-signed certificate. We set `verify=False` (controlled by `TOOL_VERIFY_TLS=0`).

### 3.5 Heuristic Safety Net — `scripts/build_baseline_submission.py`

A pure-Python decision tree that:

1. Parses `user_plane_data` with `pandas.read_csv(sep='|')`
2. Finds `t_drop` = row with max throughput-drop pct vs trailing 3-row mean (≥ 30 % drop)
3. Computes Δ_RSRP, Δ_SINR between pre-drop window and at-drop row
4. Classifies into `COVERAGE`, `INTERFERENCE`, `SCHEDULER`, or `UNKNOWN`
5. Scores each option label by keyword match against the failure mode (e.g. `INTERFERENCE` boosts options mentioning "tilt down", "A3 Offset", "azimuth")
6. Returns the top-scoring option ID (or top-2/3 for multi-answer tasks)

Validated on the full training set: **mean IoU 0.268** (single-answer 31 % exact, multi-answer partial-IoU).

The agentic agent calls `heuristic_pick(scenario)` on any LLM failure (timeout, empty `\\boxed{}`, parse error). This guarantees the agent never returns an invalid answer.

The same module also produces a standalone heuristic-only submission via `submit_now.py --no_llm`, which is our **guaranteed-floor Phase 2 submission**.

### 3.6 LoRA Self-Distillation — `scripts/distill.py` + `scripts/finetune.py`

**Distillation** (`distill.py`):

For each training scenario, prompt the running LLM with **the ground truth answer** in the system prompt and ask it to construct a CoT trace that arrives at the GT. Reject traces whose final `\\boxed{...}` doesn't match. Output: `traces/train_traces.jsonl`.

```
Prompt: "You are producing training data... Correct answer: C7. Construct
        a clean reasoning trace that arrives at exactly C7 using only the
        data provided."
```

This is a "teacher forces the answer" approach. Even when the base model is weak at choosing the right option in the first place, it can explain a known-correct answer competently. About 80 % of traces pass the GT-match filter.

**Fine-tune** (`finetune.py`):

- `peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all linear")`
- Base model in 4-bit, LoRA on top, `prepare_model_for_kbit_training`
- `max_seq_length=2048` (`LORA_MAX_SEQ_LEN`), `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `learning_rate=1e-4`, 1 epoch
- HF Trainer with `DataCollatorForLanguageModeling`
- 200-trace held-out validation split; save best checkpoint by eval loss
- Output: `training/checkpoints/run_v1/best_lora/`

The orchestrator stops `llm_server` before fine-tune (frees GPU), runs the trainer, then restarts `llm_server` with `--lora training/checkpoints/run_v1/best_lora`. The agentic agent transparently uses the LoRA-attached model from that point.

---

## 4. End-to-End Data Flow (One Scenario)

```
1. Read scenario from data/Phase_2/test.json     [local file]
2. Build rag_query from task desc + options      [agent CPU]
3. Embed query with MiniLM                       [agent CPU]
4. Cosine search over 13 526 chunks              [agent CPU, ~5 ms]
5. Build user prompt:                            [agent CPU]
   ## Reference Knowledge (top-3 retrieved)
   ## Network Configuration
   ## User-Plane Time Series
   ## Signaling Plane Events
   ## Cell-Level Traffic KPIs
   ## Measurement Reports
   ## Task / ## Options / Final answer:
6. POST /v1/chat/completions                     [HTTP local → llm_server]
   tools=[6 compute tools]
7. LLM may emit <tool_call>...</tool_call>       [llm_server → agent]
8. GET https://124.71.227.61/no/<endpoint>?...   [HTTPS remote → Zindi cloud]
   X-API-Token: ..., X-Scenario-Id: ...
9. Tool result JSON                              [cloud → agent]
10. POST /v1/chat/completions (final-answer turn)[HTTP local → llm_server]
    messages=[..., tool_msg, user_msg("emit \boxed{}")]
11. Parse \boxed{Cx} (or \boxed{Cx|Cy|Cz})       [agent CPU]
12. On any failure → heuristic_pick(scenario)    [agent CPU, 0 ms]
13. Append record to completions.jsonl           [thread-safe write]
14. Flush row to result.csv every 10 scenarios   [thread-safe write]
```

---

## 5. Configuration Reference (Env Vars)

| Env | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | HF Hub auth for dataset + model download | — (required) |
| `TOOL_BEARER_TOKEN` | Phase 2 sandbox auth (sent as both `X-API-Token` and `Authorization: Bearer`) | — (required for Phase 2) |
| `TOOL_URL` | Cloud sandbox URL | `http://localhost:7860` |
| `TOOL_VERIFY_TLS` | `0` to skip cert verify (sandbox is self-signed) | `1` |
| `TEST_FILE` | Scenario JSON to evaluate against | `data/Phase_1/test.json` |
| `MODEL_NAME` | HF repo id | `Qwen/Qwen3.5-35B-A3B` |
| `USE_VLLM` | `1` = primary vLLM path, `0` = transformers+bnb fallback | `1` |
| `TENSOR_PARALLEL_SIZE` | vLLM TP size (`1` is safest on Turing) | auto |
| `QUANT_MODE` | `bitsandbytes` / `awq` / `gptq` / `none` | `bitsandbytes` |
| `GPU_MEMORY_UTILIZATION` | vLLM GPU mem fraction | `0.85` |
| `MAX_MODEL_LEN` | vLLM context length | `8192` |
| `LLM_PER_GPU_GIB` | Transformers path: max_memory per GPU | `42` |
| `AGENT_WORKERS` | Concurrent scenario workers | `1` (or N urls) |
| `RAG_K` | Number of KB chunks to retrieve | `3` |
| `AGENT_TOOLS_ALLOWLIST` | Override 6-tool filter | unset |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (essential on Turing) | — |

---

## 6. Submission Strategy (Phase 2: 3 slots, best of 3)

| # | File | Source | Expected score |
|---|---|---|---|
| 1 | `eval/results/heuristic_baseline/result_v1_raw_zindi.csv` | Pure deterministic heuristic | ~0.30 (validated on holdout) |
| 2 | `eval/results/final_<winner>/result_v1_raw_zindi.csv` | LLM + RAG (+ LoRA if it wins holdout) — the auto-picked best config from Phase N | 0.20–0.40 |
| 3 | `eval/results/final_<winner>/result_v2_multi_recall_zindi.csv` | Same winning config, alt variant | 0.20–0.40 |

Each CSV has columns `ID,Track A,Track B` matching Zindi's sample format (`Track B` left empty since we only compete in Track A). All 500 scenarios from `data/Phase_2/test.json` are answered (no empties — heuristic ensures every row has a `Cx` or `Cx|Cy|Cz`).

---

## 7. File Manifest

```
Track A/
├── _types.py                       # Pydantic Scenario / ToolCall models (locked)
├── server.py                       # Phase 1 local tool server (unused in Phase 2)
├── main.py                         # Reference agent (we built our own)
├── utils.py                        # extract_answer, compute_score (locked)
├── logger.py
├── requirements.txt / requirements-agent.txt
├── data/
│   ├── Phase_1/{train,test}.json   # 2000 / 500 scenarios (Phase 1)
│   └── Phase_2/{test,README}       # 500 scenarios (Phase 2)
├── prompts/system_prompt.md        # Original reference prompt (we built a richer one in-code)
├── knowledge/
│   ├── raw/                        # 8 × 3GPP PDFs, 10 × arXiv PDFs, HTML primers
│   └── processed/                  # chunks.json (13526), embeddings.npy (13526,384)
├── traces/train_traces.jsonl       # Distilled CoT traces (when LoRA pipeline ran)
├── training/checkpoints/run_v1/
│   └── best_lora/                  # LoRA adapter (when fine-tune completed)
├── eval/
│   ├── results/
│   │   ├── heuristic_baseline/     # Safety-net submission CSVs
│   │   ├── agentic_holdout/        # 200-scenario validation: base LLM
│   │   ├── holdout_rag/            # 200-scenario validation: LLM + RAG
│   │   ├── holdout_lora/           # 200-scenario validation: LLM + LoRA
│   │   ├── holdout_lora_rag/       # 200-scenario validation: LLM + LoRA + RAG
│   │   └── final_<winner>/         # Phase 2 test set submission
│   └── logs/run_all/               # phase2.log, llm_server*.log
└── scripts/
    ├── llm_server.py               # transformers+bnb 4-bit OpenAI shim
    ├── llm_server_vllm.py          # vLLM OpenAI server wrapper
    ├── agentic_agent.py            # Main agent (RAG + tools + heuristic fallback, parallel)
    ├── build_baseline_submission.py # Deterministic heuristic (~0.30 floor)
    ├── submit_now.py               # Single-shot LLM+heuristic submission generator
    ├── distill.py                  # GT-anchored CoT trace synthesis from train set
    ├── finetune.py                 # peft LoRA training on distilled traces
    ├── push_lora.py                # Optional HF Hub adapter push
    ├── download_rag_docs.sh        # Scrape 3GPP/arXiv/ShareTechNote/Wikipedia
    ├── build_kb_index.py           # Chunk + MiniLM-embed → chunks.json + embeddings.npy
    ├── prefetch_model.py           # Pre-download Qwen weights into HF cache
    ├── convert_to_zindi_format.py  # ID,Track A,Track B converter
    ├── score_results.py            # Compute IoU/exact-match against ground truth
    └── run_all_today.sh            # ORCHESTRATOR — runs everything in order, resumable
```

Phase order in `run_all_today.sh`:

```
A.  venv + deps (incl. vllm, sentence-transformers, trafilatura, pypdf, ...)
B.  HF_TOKEN check + prefetch base model
B'. Phase 2 data download (huggingface_hub, handles auth + Xet)
C.  Start llm_server (vLLM first, transformers+bnb fallback)
D.  Build stratified 1800/200 holdout split
E.  Heuristic safety-net submission (~30 s)            → submission #1 ready immediately
F.  Baseline agentic holdout (LLM only, no LoRA/RAG)   → SCORE_BASE
G.  Distill 1800 train scenarios (~4-6 h)
H.  Stop llm_server, free GPU
I.  LoRA fine-tune (~1-2 h)
J.  Restart llm_server with --lora
K.  LoRA holdout (no RAG)                              → SCORE_LORA
L.  Build RAG knowledge base (scrape + index)
M.  LoRA + RAG holdout                                 → SCORE_LORA_RAG
M'. RAG-only holdout (no LoRA, for ablation)           → SCORE_RAG
N.  Pick best config from {baseline, RAG, LoRA, LoRA+RAG}
O.  Final run on data/Phase_2/test.json with winner    → submissions #2, #3
P.  Convert all outputs to Zindi format
Q.  Print upload list
```

---

## 8. Holdout Scores Observed During Development

| Config | Holdout (200 scenarios) | Notes |
|---|---|---|
| Heuristic only | **0.30** | Validated, deterministic, every scenario answered |
| LLM only (verbose 7-step prompt, max_tokens=192) | 0.18 | First LLM run on this hardware |
| LLM only (compact prompt, max_tokens=256) | 0.14 | Worse — model needed the structured guidance |
| LLM only (verbose prompt restored, max_tokens=768, X-API-Token auth fixed) | TBD (run in progress) | |
| LLM + RAG | TBD | |
| LLM + LoRA | TBD | LoRA fine-tune not yet completed |
| LLM + LoRA + RAG | TBD | |

Bottom line during development: the deterministic heuristic outperformed the LLM-as-agent on this specific dataset/hardware. The LLM-side iteration produced multiple worse-than-heuristic scores. We retained the heuristic submission as the floor.

---

## 9. Lessons Learned / Honest Limitations

**What worked well:**

- **Heuristic floor matters.** Having a deterministic fallback with ~0.30 IoU meant we always had something to submit. Several otherwise-fatal infrastructure failures (CUDA OOM, vLLM crash, SSL errors, missing data files) became logged warnings rather than zero-row submissions.
- **The agent's parallel design.** When two LLM endpoints are healthy, throughput doubles. The dispatch loop is simple and robust.
- **RAG corpus selection.** Tier-1 3GPP TS specs are the highest-signal-per-token reference — far better than generic 5G overview material.
- **GT-anchored distillation.** Forcing the teacher to produce a trace that hits a known-correct answer is a clean way to bootstrap LoRA data even when the base model is weak at the task.

**What did not work / what we'd do differently:**

- **vLLM 0.21 + Qwen3.5-35B-A3B MoE + bnb 4-bit + tensor parallel** has a weight-loading shape mismatch (`output with shape [512, 1] doesn't match the broadcast shape [512, 2048]` in `_load_w13`). The workaround is `TENSOR_PARALLEL_SIZE=1` (single GPU); the full pipeline-parallel transformers path is the more reliable fallback on Turing hardware. Future: pin a specific vLLM version validated on RTX 8000.
- **Tools chosen vs tools used.** The LLM rarely called tools in practice (`tool_calls=0` for most scenarios) — the scenario data is already inlined and the model could reason from it directly. Forcing tool calls via `tool_choice=required` or system-prompt language was tried; it slowed inference without clear accuracy gain.
- **Prompt length is a real tradeoff.** A 1000-token verbose decision tree beat a 250-token compact one on this dataset. Compressing the prompt to save tokens for output hurt accuracy more than it helped.
- **Phase 2 sandbox auth header.** The README says `Authorization: Bearer`; the server actually requires `X-API-Token`. We send both. Future participants: test endpoints with `curl` before debugging your client.
- **MoE-specific scaling.** Qwen3.5-35B-A3B is MoE with 3B active params per token — peak compute per call is modest, but memory bandwidth for expert routing is high. The model loads into ~22 GB in 4-bit but KV cache and activations push it to ~28-35 GB per inference. Sizing `GPU_MEMORY_UTILIZATION` / `LLM_PER_GPU_GIB` carefully is essential.

**Limits of this approach:**

- Performance is hardware-bound. The same code on an Ampere+ GPU (A100, H100) would let vLLM tensor-parallel work cleanly and likely 3-5× the throughput; that would change the cost-benefit of the RAG / LoRA additions.
- The heuristic ceiling is around 0.30; to meaningfully exceed it, the LLM needs to learn the (failure-mode → option-template + correct-cell-ID) mapping from labeled examples — that is what LoRA distillation is for, but it requires the full overnight pipeline to validate.

---

## 10. Reproducibility — How to Re-run This Solution

```bash
git clone https://github.com/JONAHKYAGABA/tel.git
cd tel/"Track A"

# Required environment
export HF_TOKEN=<your-hf-token>             # for Qwen download + Phase 2 dataset
export TOOL_BEARER_TOKEN=<your-zindi-token> # Track A Phase 2 token from Zindi
export TOOL_URL=https://124.71.227.61/no    # or https://120.46.145.77/no (CN region)
export TOOL_VERIFY_TLS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEST_FILE=data/Phase_2/test.json

# Optional inference tuning
export USE_VLLM=1                           # 0 to force transformers+bnb fallback
export TENSOR_PARALLEL_SIZE=1               # safest on Turing
export QUANT_MODE=bitsandbytes
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=8192
export AGENT_WORKERS=8
export LLM_PER_GPU_GIB=42

# Run the full pipeline (resumable; ~8-10 h on 2× RTX 8000)
mkdir -p eval/logs/run_all
nohup bash scripts/run_all_today.sh > eval/logs/run_all/phase2.log 2>&1 &
tail -f eval/logs/run_all/phase2.log

# Outputs:
ls eval/results/heuristic_baseline/result_v*_zindi.csv  # safety-net submission
ls eval/results/final_*/result_v*_zindi.csv             # final submission (LLM-based)
```

All hyperparameters, seeds (`SEED=42`, `PYTHONHASHSEED=42`), and dependency versions are pinned. The pipeline is reproducible end-to-end given the same hardware and tokens.

---

## License

CC-BY-SA 4.0 (per challenge rules).
