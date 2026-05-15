"""
scripts/llm_server_vllm.py — high-throughput OpenAI-compatible LLM server
backed by vLLM with tensor-parallel + continuous batching.

Compared to the bnb-4bit transformers shim in llm_server.py:
  • Tensor parallelism: both GPUs compute every step (vs pipeline parallel)
  • Continuous batching: serves N concurrent requests in one forward pass
  • PagedAttention: better KV cache memory utilization
  • Typical throughput: 5-10x for batched inference

Quantization options (pick one via env QUANT_MODE):
  • bitsandbytes — uses existing 4-bit weights. May be slow on Turing.
  • awq           — best speed on Turing, requires AWQ checkpoint
  • gptq          — also fast on Turing, requires GPTQ checkpoint
  • none          — fp16; needs both GPUs combined (~70 GB), tight

vLLM has its own OpenAI server entrypoint, so this script just exec's it
with the right args. No custom wrapping needed.

Run:
    pip install vllm
    QUANT_MODE=bitsandbytes \\
    TENSOR_PARALLEL_SIZE=2 \\
    python scripts/llm_server_vllm.py --model Qwen/Qwen3.5-35B-A3B --port 8001
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--lora", default=os.environ.get("LORA_PATH"),
                    help="Optional LoRA adapter path. vLLM accepts multi-LoRA hot-loading.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LLM_PORT", "8001")))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "2")),
                    help="Tensor-parallel size (1 = single GPU, 2 = both GPUs).")
    ap.add_argument("--gpu_mem", type=float,
                    default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")),
                    help="Fraction of each GPU vLLM may use for weights+KV (0..1).")
    ap.add_argument("--max_model_len", type=int,
                    default=int(os.environ.get("MAX_MODEL_LEN", "8192")),
                    help="Max context length (prompt+output). Bigger = more KV cache.")
    args = ap.parse_args()

    quant = os.environ.get("QUANT_MODE", "bitsandbytes").lower()

    # Verify vllm is importable
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("ERROR: vllm is not installed. Install with:", file=sys.stderr)
        print("  pip install vllm", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
        "--dtype", "float16",
        "--disable-log-requests",
    ]

    if quant == "bitsandbytes":
        # vLLM ≥0.6 supports bnb 4-bit on-the-fly
        cmd += ["--quantization", "bitsandbytes",
                "--load-format", "bitsandbytes"]
    elif quant in ("awq", "gptq"):
        cmd += ["--quantization", quant]
    elif quant in ("none", "fp16", ""):
        pass
    else:
        print(f"ERROR: unknown QUANT_MODE={quant}", file=sys.stderr)
        return 1

    if args.lora and os.path.isdir(args.lora):
        # Allow LoRA hot-loading; needs --enable-lora + --max-lora-rank matching adapter
        cmd += [
            "--enable-lora",
            "--max-lora-rank", os.environ.get("MAX_LORA_RANK", "32"),
            "--lora-modules", f"adapter={args.lora}",
        ]

    print("[vllm] launching:", " ".join(cmd), flush=True)
    # Replace current process — vLLM owns stdout/stderr now
    os.execvp(cmd[0], cmd)
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main())
