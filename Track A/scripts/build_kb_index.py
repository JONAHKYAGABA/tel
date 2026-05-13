"""
scripts/build_kb_index.py

Chunks the scraped raw .txt files in knowledge/raw/ and embeds each chunk
with sentence-transformers/all-MiniLM-L6-v2 (~80 MB, runs on CPU). Saves:

    knowledge/processed/chunks.json     # [{source, chunk_id, text}, ...]
    knowledge/processed/embeddings.npy  # float32, shape (n_chunks, 384)

Run after scrape_5g_kb.py. Idempotent — overwrites previous index.

Usage:
    pip install sentence-transformers numpy
    unset HF_HUB_OFFLINE          # so the embedding model can download
    python scripts/build_kb_index.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# If HF_HUB_OFFLINE was set by an earlier run_all.sh, the embedding model
# can't download. Clear it for this script.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.stderr.write(
        "Missing deps. Install with:\n"
        "  pip install sentence-transformers numpy\n"
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
RAW_DIR = PROJECT_DIR / "knowledge" / "raw"
PROC_DIR = PROJECT_DIR / "knowledge" / "processed"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunk parameters: paragraphs > 100 chars, hard-split anything > 1500 chars at 1200.
MIN_PARA = 100
MAX_CHUNK = 1500
SPLIT_AT = 1200


def _read_file_text(path: Path) -> str:
    """Return plain text for .txt / .html / .pdf. Returns '' on unsupported types."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".html", ".htm"):
        # Strip HTML — prefer trafilatura if available, else regex fallback
        try:
            import trafilatura  # type: ignore
            html = path.read_text(encoding="utf-8", errors="ignore")
            txt = trafilatura.extract(html, include_comments=False, include_tables=True)
            return txt or ""
        except ImportError:
            html = path.read_text(encoding="utf-8", errors="ignore")
            txt = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
            txt = re.sub(r"<style[\s\S]*?</style>", "", txt, flags=re.IGNORECASE)
            txt = re.sub(r"<[^>]+>", " ", txt)
            return re.sub(r"\s+", " ", txt)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            print(f"  [skip] pypdf not installed; cannot parse {path.name}", file=sys.stderr)
            return ""
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n\n".join(pages)
        except Exception as exc:
            print(f"  [skip] failed to parse {path.name}: {exc}", file=sys.stderr)
            return ""
    return ""


def chunk_file(path: Path) -> list[dict]:
    text = _read_file_text(path)
    if not text:
        return []

    # Strip a SOURCE: header if it's a .txt scraped file
    head, _, maybe_body = text.partition("\n\n")
    if head.startswith("SOURCE:"):
        source = head.replace("SOURCE: ", "").strip()
        body = maybe_body if maybe_body else text
    else:
        source = path.name
        body = text

    chunks: list[dict] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > MIN_PARA]
    # If no paragraph breaks (PDF text often comes back as one blob), fall back to fixed-size split
    if not paras and len(body.strip()) > MIN_PARA:
        paras = [body[i:i + MAX_CHUNK] for i in range(0, len(body), SPLIT_AT)]
    for i, p in enumerate(paras):
        if len(p) <= MAX_CHUNK:
            chunks.append({
                "source": source,
                "chunk_id": f"{path.stem}_{i}",
                "text": p,
            })
        else:
            for j in range(0, len(p), SPLIT_AT):
                chunks.append({
                    "source": source,
                    "chunk_id": f"{path.stem}_{i}_{j}",
                    "text": p[j:j + MAX_CHUNK],
                })
    return chunks


def main() -> int:
    if not RAW_DIR.exists():
        print(f"FATAL: {RAW_DIR} not found. Run scripts/download_rag_docs.sh first.",
              file=sys.stderr)
        return 1
    # Accept .txt, .md, .html/.htm, .pdf
    patterns = ("*.txt", "*.md", "*.html", "*.htm", "*.pdf")
    raw_files: list[Path] = []
    for p in patterns:
        raw_files.extend(RAW_DIR.glob(p))
    raw_files = sorted(set(raw_files))
    if not raw_files:
        print(f"FATAL: no .txt/.md/.html/.pdf in {RAW_DIR}", file=sys.stderr)
        return 1

    print(f"[index] reading {len(raw_files)} raw files from {RAW_DIR}")
    all_chunks: list[dict] = []
    for f in raw_files:
        cs = chunk_file(f)
        all_chunks.extend(cs)
        print(f"  {f.name:>40s}  ->  {len(cs):>4d} chunks")
    print(f"[index] total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("FATAL: produced 0 chunks; check raw text contents", file=sys.stderr)
        return 1

    print(f"[index] loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    print(f"[index] embedding {len(texts)} chunks (CPU is fine, ~30-90s)")
    embs = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    embs = embs.astype("float32")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROC_DIR / "embeddings.npy", embs)
    (PROC_DIR / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[index] saved:")
    print(f"  {PROC_DIR / 'chunks.json'}     ({len(all_chunks)} chunks)")
    print(f"  {PROC_DIR / 'embeddings.npy'}  ({embs.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
