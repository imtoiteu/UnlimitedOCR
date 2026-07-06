# Unlimited OCR — Local Setup & Run Guide

## Quick Start (after a clean clone)

```bash
# 1. Clone
git clone https://github.com/imtoiteu/UnlimitedOCR.git
cd UnlimitedOCR

# 2. One-time setup (creates venv, installs deps)
bash scripts/setup.sh

# 3. (Optional) Configure model path, port, etc.
cp .env.example .env    # already done by setup.sh
nano .env               # set MODEL_DIR, PORT, HF_TOKEN if needed

# 4. Health check
bash scripts/health_check.sh

# 5. Launch web UI
bash scripts/run_web.sh
# → Open http://localhost:7860 in your browser
```

---

## Prerequisites

| Tool | Minimum | Install |
|------|---------|---------|
| **uv** | any | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Python** | 3.10+ | Managed automatically by uv |
| **Git** | any | System package manager |
| **NVIDIA GPU** | Optional | Required for fast inference |

> **macOS Apple Silicon (M1/M2/M3/M4):** Works via MPS (Apple Metal).  
> Inference is slower than CUDA but fully functional for testing.

---

## Environment Variables (`.env`)

Copy `.env.example` to `.env` and edit as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `baidu/Unlimited-OCR` | HF model ID or local path |
| `PORT` | `7860` | Gradio web UI port |
| `OUTPUT_DIR` | `./outputs` | Where OCR results are saved |
| `CUDA_DEVICE` | `0` | GPU index (CUDA only) |
| `HF_TOKEN` | _(empty)_ | HuggingFace token (if model is gated) |

### Using a local model copy

If you've pre-downloaded the model weights:

```bash
# In .env:
MODEL_DIR=/path/to/local/Unlimited-OCR
```

To download manually:

```bash
source .venv/bin/activate
huggingface-cli download baidu/Unlimited-OCR \
    --local-dir ./models/Unlimited-OCR \
    --local-dir-use-symlinks False
# Then set MODEL_DIR=./models/Unlimited-OCR in .env
```

---

## Web UI Usage

1. **Upload** a PNG/JPG/WEBP/BMP image or a PDF file.
2. **Select config** (single images only):
   - `gundam` — 640 px crop mode — best for dense/complex documents
   - `base` — 1024 px no-crop — best for clean, simple layouts
3. Click **Run OCR**.
4. Results appear in the **Markdown Preview** and **Raw Text** tabs.
5. All results are auto-saved to `outputs/<timestamp>/`.

---

## Inference Backends

### Transformers (used by the web UI)

Works on CPU, Apple MPS, and CUDA. Selected automatically.

```python
from transformers import AutoModel, AutoTokenizer
# Single image
model.infer(tokenizer, prompt="<image>document parsing.", ...)
# Multi-page / PDF
model.infer_multi(tokenizer, prompt="<image>Multi page parsing.", ...)
```

### SGLang (batch / high-throughput — NVIDIA + Linux only)

```bash
# Start server
python -m sglang.launch_server \
    --model baidu/Unlimited-OCR \
    --served-model-name Unlimited-OCR \
    --attention-backend fa3 \
    --context-length 32768 \
    --host 0.0.0.0 --port 10000

# Run batch inference
python infer.py \
    --image_dir ./examples/images \
    --output_dir ./outputs \
    --concurrency 8 \
    --image_mode gundam
```

---

## GPU / Device Summary

| Platform | Device | Speed | Notes |
|----------|--------|-------|-------|
| Linux + NVIDIA | CUDA | ⚡ Fast | Recommended for production |
| macOS Apple Silicon | MPS (Metal) | 🟡 Medium | Good for testing |
| Any | CPU | 🐢 Slow | Fallback, not recommended for large docs |

---

## Troubleshooting

### Model download fails / HF 401 error

```bash
# Login to HuggingFace
source .venv/bin/activate
huggingface-cli login
# Or set HF_TOKEN in .env
```

### `torch` not found after setup

```bash
bash scripts/setup.sh   # re-run to reinstall
```

### Port already in use

```bash
PORT=8080 bash scripts/run_web.sh
```

### Out of memory on CPU/MPS

- Try uploading smaller images (< 2000 px)
- For PDFs, use a lower DPI (edit `app.py`: change `dpi=200` to `dpi=150`)

### SGLang fails on macOS

SGLang requires Linux + NVIDIA CUDA. Use the Transformers-backed web UI (`bash scripts/run_web.sh`) on macOS.

---

## File Structure

```
UnlimitedOCR/
├── app.py                  # Gradio web UI (Transformers backend)
├── infer.py                # SGLang batch inference CLI
├── scripts/
│   ├── setup.sh            # One-time environment setup
│   ├── run_web.sh          # Launch web UI
│   └── health_check.sh     # Verify environment
├── .env.example            # Config template (copy to .env)
├── .env                    # Local config (git-ignored)
├── outputs/                # OCR results (git-ignored)
├── log/                    # Server logs (git-ignored)
├── wheel/                  # SGLang wheel for Linux/CUDA
└── README_RUN.md           # This file
```

---

## Known Limitations

- **Model size**: ~7–14 GB. First run downloads automatically; ensure sufficient
  disk space and a stable internet connection.
- **MPS (Apple Silicon)**: Some operations may fall back to CPU; inference is
  slower than CUDA but produces identical results.
- **SGLang wheel** (`wheel/sglang-*.whl`): Pre-built for Linux. Ignored on macOS.
- **`assets/long-horizon-ocr.gif`** is 82 MB and excluded from the repo push
  (too large for standard GitHub hosting).
