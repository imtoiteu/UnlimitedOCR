#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_web.sh — Start the Unlimited-OCR Gradio web UI
#
# Usage:
#   bash scripts/run_web.sh
#   PORT=8080 bash scripts/run_web.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'
info()    { echo -e "${BLU}[INFO]${RST}  $*"; }
success() { echo -e "${GRN}[OK]${RST}    $*"; }
warn()    { echo -e "${YLW}[WARN]${RST}  $*"; }
error()   { echo -e "${RED}[ERROR]${RST} $*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Check venv ────────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    error ".venv not found. Run setup first:"
    error "  bash scripts/setup.sh"
    exit 1
fi
source .venv/bin/activate

# ── Load .env (if present) ─────────────────────────────────────────────────────
if [ -f ".env" ]; then
    # Export only non-comment, non-empty lines
    set -o allexport
    # shellcheck source=/dev/null
    source .env
    set +o allexport
    info "Loaded configuration from .env"
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
export MODEL_DIR="${MODEL_DIR:-baidu/Unlimited-OCR}"
export PORT="${PORT:-7860}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$OUTPUT_DIR" log

# ── Print launch info ─────────────────────────────────────────────────────────
echo ""
echo -e "${CYN}╔══════════════════════════════════════════╗${RST}"
echo -e "${CYN}║         Unlimited-OCR Web UI             ║${RST}"
echo -e "${CYN}╠══════════════════════════════════════════╣${RST}"
echo -e "${CYN}║${RST}  Model : ${MODEL_DIR}  "
echo -e "${CYN}║${RST}  URL   : http://localhost:${PORT}"
echo -e "${CYN}║${RST}  Output: ${OUTPUT_DIR}"
echo -e "${CYN}║${RST}  (First run downloads the model — be patient)"
echo -e "${CYN}╚══════════════════════════════════════════╝${RST}"
echo ""

# ── Launch Gradio ──────────────────────────────────────────────────────────────
exec python app.py
