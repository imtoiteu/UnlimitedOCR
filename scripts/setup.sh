#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/setup.sh — Unlimited-OCR reproducible environment setup
#
# Usage:
#   bash scripts/setup.sh
#
# What it does:
#   1. Verifies prerequisites (uv, Python ≥ 3.10)
#   2. Creates .venv with Python 3.12 via uv
#   3. Installs all Python dependencies (Transformers path)
#   4. Optionally installs SGLang wheel (Linux + CUDA only)
#   5. Detects CUDA / MPS / CPU and prints device summary
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'
info()    { echo -e "${BLU}[INFO]${RST}  $*"; }
success() { echo -e "${GRN}[OK]${RST}    $*"; }
warn()    { echo -e "${YLW}[WARN]${RST}  $*"; }
error()   { echo -e "${RED}[ERROR]${RST} $*" >&2; }

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
info "Repository root: $REPO_ROOT"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
info "Checking prerequisites…"

if ! command -v uv &>/dev/null; then
    error "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
success "uv $(uv --version)"

# ── 2. Create virtual environment ─────────────────────────────────────────────
info "Creating .venv with Python 3.12…"
if [ -d ".venv" ]; then
    warn ".venv already exists — skipping creation (delete it to recreate)"
else
    uv venv --python 3.12
    success ".venv created"
fi

# Activate for the rest of this script
source .venv/bin/activate
PYTHON=".venv/bin/python"
success "Python: $(python --version)"

# ── 3. Detect compute device ──────────────────────────────────────────────────
info "Detecting compute device…"

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
HAS_CUDA=false
HAS_MPS=false

if command -v nvcc &>/dev/null; then
    HAS_CUDA=true
    CUDA_VERSION="$(nvcc --version 2>&1 | grep -oP 'release \K[\d.]+')"
    success "CUDA found: $CUDA_VERSION"
elif [ "$PLATFORM" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    HAS_MPS=true
    success "Apple Silicon (MPS / Metal) detected"
else
    warn "No CUDA or MPS detected — will run on CPU (slow for large documents)"
fi

# ── 4. Install core Python dependencies ───────────────────────────────────────
info "Installing core Python packages…"

# Pin torch to a version compatible with Python 3.12
if $HAS_CUDA; then
    # Install CUDA-enabled PyTorch (CUDA 12.x)
    uv pip install \
        "torch==2.5.1" \
        "torchvision==0.20.1" \
        --index-url https://download.pytorch.org/whl/cu124
else
    # CPU / MPS build
    uv pip install \
        "torch==2.5.1" \
        "torchvision==0.20.1"
fi

uv pip install \
    "transformers==4.57.1" \
    "Pillow>=10.0.0" \
    "einops>=0.8.0" \
    "addict>=2.4.0" \
    "easydict>=1.13" \
    "pymupdf>=1.24.0" \
    "psutil>=5.9.0" \
    "accelerate>=0.27.0" \
    "gradio>=5.0.0" \
    "python-dotenv>=1.0.0" \
    "huggingface-hub>=0.23.0" \
    "safetensors>=0.4.0" \
    "requests>=2.31.0"

success "Core packages installed"

# ── 5. Install SGLang wheel (Linux + CUDA only) ───────────────────────────────
WHEEL_PATH="$REPO_ROOT/wheel/sglang-0.0.0.dev11416+g92e8bb79e-py3-none-any.whl"
if $HAS_CUDA && [ "$PLATFORM" = "Linux" ]; then
    if [ -f "$WHEEL_PATH" ]; then
        info "Installing SGLang wheel (CUDA + Linux detected)…"
        uv pip install "$WHEEL_PATH"
        uv pip install "kernels==0.11.7"
        success "SGLang installed"
    else
        warn "SGLang wheel not found at $WHEEL_PATH — skipping batch inference"
    fi
else
    warn "SGLang skipped (requires Linux + CUDA). Gradio UI uses Transformers path."
fi

# ── 6. Create necessary directories ──────────────────────────────────────────
info "Creating output directories…"
mkdir -p outputs log
success "outputs/ and log/ ready"

# ── 7. Copy .env.example → .env if not present ───────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    success ".env created from .env.example (edit it to configure model path etc.)"
elif [ -f ".env" ]; then
    info ".env already exists — leaving untouched"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYN}╔══════════════════════════════════════════╗${RST}"
echo -e "${CYN}║       Setup complete! Quick start:       ║${RST}"
echo -e "${CYN}╠══════════════════════════════════════════╣${RST}"
echo -e "${CYN}║${RST}  bash scripts/run_web.sh                ${CYN}║${RST}"
echo -e "${CYN}║${RST}  Open → http://localhost:7860            ${CYN}║${RST}"
echo -e "${CYN}╠══════════════════════════════════════════╣${RST}"
if $HAS_CUDA; then
echo -e "${CYN}║${RST}  GPU: ${GRN}CUDA $CUDA_VERSION${RST}                       ${CYN}║${RST}"
elif $HAS_MPS; then
echo -e "${CYN}║${RST}  GPU: ${YLW}Apple MPS (Metal)${RST}                ${CYN}║${RST}"
else
echo -e "${CYN}║${RST}  GPU: ${RED}None (CPU only — slow)${RST}           ${CYN}║${RST}"
fi
echo -e "${CYN}╚══════════════════════════════════════════╝${RST}"
echo ""
