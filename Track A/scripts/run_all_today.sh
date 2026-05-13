#!/usr/bin/env bash
# scripts/run_all_today.sh
#
# THE ONE COMMAND. Builds everything end-to-end on a fresh RunPod box.
# Resumable — re-run any time and it skips finished steps.
#
# Phases:
#   A.  venv + ALL deps (torch, transformers, peft, bnb, fastapi, sentence-transformers, trafilatura, pandas, ...)
#   B.  HF_TOKEN check + prefetch base model (~30 min, ~67 GB)
#   C.  Start llm_server (port 8001) + locked server.py (port 7860)
#   D.  Build stratified 1800/200 holdout split
#   E.  Heuristic safety-net submission (~30 sec) -> Zindi-ready CSV
#   F.  Agentic baseline holdout (LLM only, no LoRA, no RAG)
#   G.  Distill 1800 train traces  (~4-6 hours)
#   H.  Stop llm_server, free GPU
#   I.  LoRA fine-tune              (~1-2 hours)
#   J.  Restart llm_server with --lora
#   K.  Agentic holdout with LoRA   (no RAG yet)
#   L.  Scrape 5G knowledge base + build embedding index (~10 min)
#   M.  Agentic holdout with LoRA + RAG
#   N.  Decide best config (baseline / LoRA / LoRA+RAG / heuristic floor)
#   O.  Final run on data/Phase_1/test.json with winning config
#   P.  Convert all outputs to Zindi format (ID, Track A, Track B)
#   Q.  Print upload list
#
# Usage:
#     cd /workspace/tel/"Track A"
#     export HF_TOKEN=hf_PASTE_YOUR_TOKEN_HERE
#     nohup bash scripts/run_all_today.sh > eval/logs/run_all/today.log 2>&1 &
#     echo "pid=$!"
#     tail -f eval/logs/run_all/today.log
#
# Optional skips (override via env):
#     SKIP_LORA=1     skip distill + finetune (use base model only)
#     SKIP_RAG=1      skip knowledge-base scrape/index
#     SKIP_HOLDOUT=1  straight to final (use cached best config)
#     SKIP_FINAL=1    stop after holdout scoring
#     SKIP_INSTALL=1  trust that deps are present (skip pip install)
#
# Total runtime end-to-end on RTX A5000 24GB: ~8-10 hours
# (heuristic submission appears in ~3 min; LoRA + RAG complete by hour 10).

set -uo pipefail
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ============ config ===============================================
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
TOOL_PORT="${TOOL_PORT:-7860}"
# Phase 2: set TOOL_URL to Zindi's cloud endpoint, e.g.
#   export TOOL_URL=https://124.71.227.61/no
#   export TOOL_BEARER_TOKEN=<your-token>
# Default = local server.py (Phase 1 style).
TOOL_URL="${TOOL_URL:-http://localhost:$TOOL_PORT}"
TEST_FILE="${TEST_FILE:-data/Phase_1/test.json}"
TRAIN_FOLD="${TRAIN_FOLD:-data/local_split/train_1800.json}"
HOLDOUT="${HOLDOUT:-data/local_split/holdout_200.json}"
TRACES="${TRACES:-traces/train_traces.jsonl}"
LORA_PARENT="${LORA_PARENT:-training/checkpoints/run_v1}"
LORA_DIR="$LORA_PARENT/best_lora"
LLM_LOG="eval/logs/run_all/llm_server.log"
TOOL_LOG="eval/logs/run_all/tool_server.log"
KB_DIR="knowledge"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_RAG="${SKIP_RAG:-0}"
SKIP_HOLDOUT="${SKIP_HOLDOUT:-0}"
SKIP_FINAL="${SKIP_FINAL:-0}"
DISTILL_LIMIT="${DISTILL_LIMIT:-1800}"

# LoRA hyperparams tuned for 24 GB A5000
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_MAX_SEQ_LEN="${LORA_MAX_SEQ_LEN:-2048}"
LORA_EPOCHS="${LORA_EPOCHS:-1}"
LORA_BATCH="${LORA_BATCH:-1}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-16}"
LORA_LR="${LORA_LR:-1e-4}"

mkdir -p eval/logs/run_all eval/results traces "$LORA_PARENT" \
         data/local_split "$KB_DIR/raw" "$KB_DIR/processed"

# ============ ui helpers ===========================================
c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }
step()    { echo; c_blue "========================================================"; c_blue "  $1"; c_blue "========================================================"; }

wait_health() {
    local deadline=$(( $(date +%s) + ${2:-900} ))
    while ! curl -sf "$1" 2>/dev/null | grep -q '"status":"ok"'; do
        [ "$(date +%s)" -gt "$deadline" ] && return 1
        printf "."; sleep 5
    done
    echo " ready"; return 0
}

start_llm_server() {
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python scripts/llm_server.py --model "$MODEL_NAME" --port "$LLM_PORT" "$@" \
        > "$LLM_LOG" 2>&1 &
    echo "  llm pid=$!"
    wait_health "http://localhost:$LLM_PORT/health" 1200 || return 1
}

ensure_tool_server() {
    # If TOOL_URL is remote (Phase 2 cloud), don't start a local server.py.
    case "$TOOL_URL" in
        http://localhost*|http://127.0.0.1*)
            ;;
        *)
            # remote — just verify reachability (auth header may be required, so 401 is OK)
            local code
            code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 \
                ${TOOL_BEARER_TOKEN:+-H "Authorization: Bearer $TOOL_BEARER_TOKEN"} \
                -k "$TOOL_URL/health" 2>/dev/null || echo 000)
            if [ "$code" = "200" ] || [ "$code" = "401" ] || [ "$code" = "403" ]; then
                c_yel "  using remote TOOL_URL=$TOOL_URL  (http=$code)"
                return 0
            fi
            c_red "  remote TOOL_URL=$TOOL_URL unreachable (http=$code)"
            return 1
            ;;
    esac
    if curl -sf "$TOOL_URL/health" 2>/dev/null | grep -q '"status":"ok"'; then
        return 0
    fi
    pkill -f "python server.py" 2>/dev/null || true
    sleep 2
    DATA_SPLIT=test nohup python server.py > "$TOOL_LOG" 2>&1 &
    echo "  tool pid=$!"
    wait_health "$TOOL_URL/health" 60
}

zindi_convert() {
    python -c "
import pandas as pd
df = pd.read_csv('$1', dtype=str).fillna('')
df = df.rename(columns={'scenario_id':'ID','answers':'Track A'})
df['Track B'] = ''
df = df[['ID','Track A','Track B']]
df.to_csv('$2', index=False)
print('  ${2##*/}:', len(df), 'rows')"
}

extract_score() {
    grep -m1 "mean   :" "$1" 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1
}

# ============ A. venv + dependencies ==============================
step "A. venv + dependencies"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    c_yel "  created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$SKIP_INSTALL" = "1" ]; then
    c_yel "  SKIP_INSTALL=1 — trusting current deps"
elif python -c "import torch, transformers, peft, bitsandbytes, fastapi, pandas, sentence_transformers, trafilatura, datasets" >/dev/null 2>&1; then
    c_yel "  all deps already present"
else
    echo "  upgrading pip"
    pip install --upgrade -q pip wheel setuptools

    echo "  installing locked requirements"
    pip install -q -r requirements.txt 2>&1 | tail -3 || true

    echo "  installing agent + LLM serving deps"
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
        "uvicorn[standard]" python-multipart "fastapi>=0.110" pandas 2>&1 | tail -2

    echo "  installing torch / transformers / peft / bnb (slow first time, 5-10 min)"
    pip install -q "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=1.0.0" \
        "bitsandbytes>=0.44.0" "peft>=0.13.0" "datasets>=3.0.0" \
        safetensors sentencepiece protobuf "huggingface_hub>=0.25" hf_transfer 2>&1 | tail -2

    echo "  installing RAG deps"
    pip install -q sentence-transformers numpy trafilatura beautifulsoup4 pypdf 2>&1 | tail -2

    c_green "  deps installed"
fi

# ============ B. HF token + model prefetch =========================
step "B. HF_TOKEN check + prefetch $MODEL_NAME"
if [ -f .env ] && [ -z "${HF_TOKEN:-}" ]; then
    set -a; source .env; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    c_red "  HF_TOKEN not set. export HF_TOKEN=hf_... and rerun."
    exit 1
fi
SAFE_NAME="models--${MODEL_NAME//\//--}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_DIR="$HF_HOME/hub/$SAFE_NAME"
if [ -d "$CACHE_DIR/blobs" ] && [ "$(du -sb "$CACHE_DIR" | cut -f1)" -ge "$((40 * 1024 * 1024 * 1024))" ]; then
    c_yel "  cache present ($(du -sh "$CACHE_DIR" | cut -f1)) — skipping"
else
    python scripts/prefetch_model.py || { c_red "prefetch failed"; exit 1; }
    c_green "  cache: $(du -sh "$CACHE_DIR" | cut -f1)"
fi

# ============ C. start servers =====================================
step "C. Start llm_server on :$LLM_PORT and tool server on :$TOOL_PORT"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_yel "  llm_server already healthy"
else
    start_llm_server || { c_red "llm_server failed"; tail -n 40 "$LLM_LOG" >&2; exit 1; }
fi
ensure_tool_server || c_yel "  tool server unavailable (agent will run without tool execution)"

# ============ D. holdout split =====================================
step "D. Build stratified 1800/200 holdout"
if [ -f "$TRAIN_FOLD" ] && [ -f "$HOLDOUT" ]; then
    c_yel "  exists — skipping"
else
    python -c "
import json, random, os
from collections import defaultdict
random.seed(42)
t = json.load(open('data/Phase_1/train.json'))
buckets = defaultdict(list)
for s in t:
    k = (s.get('tag','single-answer'),
         (s.get('context',{}).get('wireless_network_information') or {}).get('num_base_stations','4'))
    buckets[k].append(s)
train, hold = [], []
for k, sc in buckets.items():
    random.shuffle(sc)
    n = max(1, len(sc)*200//2000)
    hold.extend(sc[:n]); train.extend(sc[n:])
os.makedirs('data/local_split', exist_ok=True)
json.dump(train, open('$TRAIN_FOLD','w'))
json.dump(hold,  open('$HOLDOUT','w'))
print(f'  train={len(train)} holdout={len(hold)}')"
fi

# ============ E. heuristic safety-net ==============================
step "E. Heuristic safety-net submission (~30 sec)"
if [ -f eval/results/heuristic_baseline/result_v1_raw_zindi.csv ]; then
    c_yel "  exists — skipping"
else
    rm -rf eval/results/heuristic_baseline
    python scripts/submit_now.py \
        --test_file "$TEST_FILE" \
        --out_dir   eval/results/heuristic_baseline \
        --no_llm 2>&1 | tail -8
    for v in v1_raw v2_multi_recall v3_insurance; do
        zindi_convert "eval/results/heuristic_baseline/result_${v}.csv" \
                      "eval/results/heuristic_baseline/result_${v}_zindi.csv"
    done
fi

# ============ F. baseline agentic holdout ==========================
SCORE_BASE=""
if [ "$SKIP_HOLDOUT" = "1" ]; then
    c_yel "F. SKIP_HOLDOUT=1"
else
    step "F. Agentic baseline holdout (LLM, no LoRA, no RAG)"
    if [ ! -f eval/results/agentic_holdout.log ] || \
       [ ! -f eval/results/agentic_holdout/result.csv ] || \
       [ "$(wc -l < eval/results/agentic_holdout/result.csv 2>/dev/null || echo 0)" -lt 200 ]; then
        rm -rf eval/results/agentic_holdout
        python scripts/agentic_agent.py \
            --test_file "$HOLDOUT" \
            --out_dir   eval/results/agentic_holdout \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "$TOOL_URL" \
            --max_tokens 384 --max_tool_calls 2 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/agentic_holdout.log
    fi
    SCORE_BASE=$(extract_score eval/results/agentic_holdout.log)
    c_green "  baseline holdout: ${SCORE_BASE:-?}"
fi

# ============ G. distill ===========================================
DISTILL_OK=0
if [ "$SKIP_LORA" = "1" ]; then
    c_yel "G. SKIP_LORA=1 — skipping distill"
else
    step "G. Distill 1800 train traces (~4-6 hours)"
    if [ -f "$TRACES" ] && [ "$(wc -l < "$TRACES")" -ge 800 ]; then
        c_yel "  $TRACES has $(wc -l < "$TRACES") traces — skipping"
        DISTILL_OK=1
    else
        BEFORE=0; [ -f "$TRACES" ] && BEFORE=$(wc -l < "$TRACES")
        python scripts/distill.py \
            --train_file "$TRAIN_FOLD" \
            --output    "$TRACES" \
            --model_url "http://localhost:$LLM_PORT/v1" \
            --model_name "$MODEL_NAME" \
            --max_samples "$DISTILL_LIMIT" || c_red "  distill failed"
        AFTER=$(wc -l < "$TRACES")
        c_green "  traces: $BEFORE -> $AFTER"
        [ "$AFTER" -ge 200 ] && DISTILL_OK=1 || c_red "  too few traces — skipping fine-tune"
    fi
fi

# ============ H+I. LoRA fine-tune ==================================
LORA_AVAILABLE=0
if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    c_yel "I. LoRA already at $LORA_DIR — skipping"
    LORA_AVAILABLE=1
elif [ "$SKIP_LORA" = "1" ] || [ "$DISTILL_OK" != "1" ]; then
    c_yel "I. skipping fine-tune"
else
    step "H. Stop llm_server to free GPU"
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 8

    step "I. LoRA fine-tune (~1-2 hours)"
    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       python scripts/finetune.py \
           --traces "$TRACES" \
           --output_dir "$LORA_PARENT" \
           --base_model "$MODEL_NAME" \
           --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
           --lora_dropout "$LORA_DROPOUT" \
           --max_seq_length "$LORA_MAX_SEQ_LEN" \
           --epochs "$LORA_EPOCHS" \
           --per_device_batch_size "$LORA_BATCH" \
           --grad_accum "$LORA_GRAD_ACCUM" \
           --lr "$LORA_LR"; then
        if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
            LORA_AVAILABLE=1
            c_green "  saved $LORA_DIR"
        fi
    else
        c_red "  fine-tune failed (try LORA_MAX_SEQ_LEN=1024 LORA_R=4)"
    fi
fi

# ============ J. restart llm_server (with LoRA if available) ========
step "J. Restart llm_server"
if [ "$LORA_AVAILABLE" = "1" ]; then
    start_llm_server --lora "$LORA_DIR" || { c_red "lora server failed"; exit 1; }
else
    start_llm_server || { c_red "base server restart failed"; exit 1; }
fi
ensure_tool_server || true

# ============ K. holdout WITH LoRA =================================
SCORE_LORA=""
if [ "$LORA_AVAILABLE" = "1" ] && [ "$SKIP_HOLDOUT" != "1" ]; then
    step "K. Agentic holdout with LoRA (no RAG)"
    if [ ! -f eval/results/holdout_lora.log ] || \
       [ ! -f eval/results/holdout_lora/result.csv ] || \
       [ "$(wc -l < eval/results/holdout_lora/result.csv 2>/dev/null || echo 0)" -lt 200 ]; then
        rm -rf eval/results/holdout_lora
        python scripts/agentic_agent.py \
            --test_file "$HOLDOUT" \
            --out_dir   eval/results/holdout_lora \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "$TOOL_URL" \
            --max_tokens 384 --max_tool_calls 2 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/holdout_lora.log
    fi
    SCORE_LORA=$(extract_score eval/results/holdout_lora.log)
    c_green "  LoRA holdout: ${SCORE_LORA:-?}"
fi

# ============ L. RAG knowledge base ================================
RAG_AVAILABLE=0
if [ "$SKIP_RAG" = "1" ]; then
    c_yel "L. SKIP_RAG=1"
elif [ -f "$KB_DIR/processed/embeddings.npy" ] && [ -f "$KB_DIR/processed/chunks.json" ]; then
    c_yel "L. KB already built ($(python -c "import json;print(len(json.load(open('$KB_DIR/processed/chunks.json'))))") chunks)"
    RAG_AVAILABLE=1
else
    step "L. Scrape 5G knowledge base + build embedding index"
    if [ -f scripts/scrape_5g_kb.py ]; then
        python scripts/scrape_5g_kb.py || c_yel "  scrape failed (some URLs may be blocked) — continuing"
    fi
    # Build index from whatever is in knowledge/raw (could be PDFs you wgetted manually)
    if [ -d "$KB_DIR/raw" ] && [ "$(ls -A "$KB_DIR/raw" 2>/dev/null | wc -l)" -gt 0 ]; then
        python scripts/build_kb_index.py && RAG_AVAILABLE=1 || c_red "  index build failed"
    else
        c_yel "  $KB_DIR/raw is empty — no docs to index; skipping RAG"
    fi
fi

# ============ M. holdout WITH LoRA + RAG ==========================
SCORE_LORA_RAG=""
if [ "$LORA_AVAILABLE" = "1" ] && [ "$RAG_AVAILABLE" = "1" ] && [ "$SKIP_HOLDOUT" != "1" ]; then
    step "M. Agentic holdout with LoRA + RAG"
    # NOTE: agentic_agent.py doesn't have --use_rag yet; falls through transparently.
    # If you've wired it, add the flag. For now we score the same as M=K.
    if [ ! -f eval/results/holdout_lora_rag.log ] || \
       [ ! -f eval/results/holdout_lora_rag/result.csv ] || \
       [ "$(wc -l < eval/results/holdout_lora_rag/result.csv 2>/dev/null || echo 0)" -lt 200 ]; then
        rm -rf eval/results/holdout_lora_rag
        python scripts/agentic_agent.py \
            --test_file "$HOLDOUT" \
            --out_dir   eval/results/holdout_lora_rag \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "$TOOL_URL" \
            --max_tokens 384 --max_tool_calls 2 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/holdout_lora_rag.log
    fi
    SCORE_LORA_RAG=$(extract_score eval/results/holdout_lora_rag.log)
    c_green "  LoRA+RAG holdout: ${SCORE_LORA_RAG:-?}"
fi

# ============ N. pick best config ==================================
step "N. Pick best config"
BEST_LABEL="baseline"; BEST_SCORE="$SCORE_BASE"
choose() {
    [ -z "$2" ] && return
    if [ -z "$BEST_SCORE" ] || python -c "import sys; sys.exit(0 if float('$2') > float('$BEST_SCORE') else 1)"; then
        BEST_LABEL="$1"; BEST_SCORE="$2"
    fi
}
choose "baseline" "$SCORE_BASE"
choose "LoRA"     "$SCORE_LORA"
choose "LoRA+RAG" "$SCORE_LORA_RAG"

echo "  baseline   : ${SCORE_BASE:-?}"
echo "  LoRA       : ${SCORE_LORA:-?}"
echo "  LoRA+RAG   : ${SCORE_LORA_RAG:-?}"
c_green "  WINNER -> $BEST_LABEL (score=${BEST_SCORE:-?})"

# Ensure server matches winning config
case "$BEST_LABEL" in
    baseline)
        if curl -s "http://localhost:$LLM_PORT/health" | grep -q '+lora'; then
            start_llm_server || c_red "  could not switch to base"
        fi ;;
    LoRA*)
        if ! curl -s "http://localhost:$LLM_PORT/health" | grep -q '+lora'; then
            start_llm_server --lora "$LORA_DIR" || c_red "  could not switch to LoRA"
        fi ;;
esac

# ============ O. final run on test set =============================
case "$BEST_LABEL" in
    baseline) FINAL_DIR="eval/results/final_baseline" ;;
    LoRA)     FINAL_DIR="eval/results/final_lora" ;;
    LoRA+RAG) FINAL_DIR="eval/results/final_lora_rag" ;;
    *)        FINAL_DIR="eval/results/final" ;;
esac

if [ "$SKIP_FINAL" = "1" ]; then
    c_yel "O. SKIP_FINAL=1 — not running on test set"
else
    step "O. Final run on $TEST_FILE  (output: $FINAL_DIR)"
    if [ -f "$FINAL_DIR/result.csv" ] && \
       [ "$(wc -l < "$FINAL_DIR/result.csv")" -ge 500 ]; then
        c_yel "  $FINAL_DIR/result.csv already complete — skipping"
    else
        rm -rf "$FINAL_DIR"
        python scripts/agentic_agent.py \
            --test_file "$TEST_FILE" \
            --out_dir   "$FINAL_DIR" \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "$TOOL_URL" \
            --max_tokens 384 --max_tool_calls 2 \
            --scenario_timeout_s 120 2>&1 | tee "${FINAL_DIR}.log"
    fi
fi

# ============ P. convert to Zindi ==================================
step "P. Convert outputs to Zindi format (ID,Track A,Track B)"
if [ -d "$FINAL_DIR" ]; then
    for v in v1_raw v2_multi_recall v3_insurance; do
        if [ -f "$FINAL_DIR/result_${v}.csv" ]; then
            zindi_convert "$FINAL_DIR/result_${v}.csv" "$FINAL_DIR/result_${v}_zindi.csv"
        fi
    done
fi

# ============ Q. summary ===========================================
step "Q. DONE"
echo
echo "Holdout scores:"
echo "  baseline   : ${SCORE_BASE:-?}"
echo "  LoRA       : ${SCORE_LORA:-?}"
echo "  LoRA+RAG   : ${SCORE_LORA_RAG:-?}"
echo "  WINNER     : $BEST_LABEL (${BEST_SCORE:-?})"
echo
c_green "Phase 2 submission candidates (best of 3 counted by Zindi):"
echo "  1) $FINAL_DIR/result_v1_raw_zindi.csv            # winner: $BEST_LABEL"
echo "  2) $FINAL_DIR/result_v2_multi_recall_zindi.csv   # winner: $BEST_LABEL (alt)"
echo "  3) eval/results/heuristic_baseline/result_v1_raw_zindi.csv  # safety net"
echo
ls -la "$FINAL_DIR/result_"*_zindi.csv 2>/dev/null
ls -la eval/results/heuristic_baseline/result_v1_raw_zindi.csv 2>/dev/null
