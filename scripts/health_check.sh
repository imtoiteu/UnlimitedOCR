#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/health_check.sh — Verify the Unlimited-OCR environment
#
# Checks: Python version, venv, key packages, CUDA/MPS, HF model access.
# Exits 0 if all critical checks pass; exits 1 if any critical check fails.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'

PASS=0; WARN=0; FAIL=0

pass() { echo -e "  ${GRN}✔${RST}  $*"; ((PASS++)); }
warn() { echo -e "  ${YLW}⚠${RST}  $*"; ((WARN++)); }
fail() { echo -e "  ${RED}✘${RST}  $*"; ((FAIL++)); }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo -e "${CYN}══════════════════════════════════════════${RST}"
echo -e "${CYN}   Unlimited-OCR  Health Check            ${RST}"
echo -e "${CYN}══════════════════════════════════════════${RST}"
echo ""

# ── 1. uv ─────────────────────────────────────────────────────────────────────
echo "[ uv ]"
if command -v uv &>/dev/null; then
    pass "uv $(uv --version)"
else
    fail "uv not found — install from https://docs.astral.sh/uv/"
fi

# ── 2. Virtual environment ────────────────────────────────────────────────────
echo ""
echo "[ Virtual Environment ]"
if [ -f ".venv/bin/python" ]; then
    PYVER="$(.venv/bin/python --version 2>&1)"
    pass ".venv exists ($PYVER)"
    source .venv/bin/activate
    PYTHON=".venv/bin/python"
else
    fail ".venv not found — run: bash scripts/setup.sh"
    echo ""
    echo -e "${RED}Critical checks failed. Run setup first.${RST}"
    exit 1
fi

# Check Python version ≥ 3.10
PY_MAJOR=$(.venv/bin/python -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(.venv/bin/python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
    pass "Python $PY_MAJOR.$PY_MINOR ≥ 3.10"
else
    fail "Python $PY_MAJOR.$PY_MINOR < 3.10 (need ≥ 3.10)"
fi

# ── 3. Core imports ───────────────────────────────────────────────────────────
echo ""
echo "[ Python Packages ]"

check_import() {
    local pkg="$1"; local label="${2:-$1}"
    if python -c "import $pkg" 2>/dev/null; then
        VER=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null || echo "?")
        pass "$label $VER"
    else
        fail "$label not importable — run: bash scripts/setup.sh"
    fi
}

check_import torch
check_import torchvision
check_import transformers
check_import PIL "Pillow"
check_import gradio
check_import fitz "PyMuPDF (fitz)"
check_import einops
check_import addict
check_import easydict
check_import dotenv "python-dotenv"
check_import huggingface_hub

# ── 4. GPU / Device ───────────────────────────────────────────────────────────
echo ""
echo "[ Compute Device ]"

DEVICE_REPORT=$(python -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f'cuda:{n}:' + ','.join(names))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps:Apple Silicon')
else:
    print('cpu:none')
" 2>/dev/null)

DEVICE_TYPE="${DEVICE_REPORT%%:*}"
DEVICE_INFO="${DEVICE_REPORT#*:}"

case "$DEVICE_TYPE" in
    cuda)
        pass "CUDA available — $DEVICE_INFO"
        ;;
    mps)
        pass "Apple MPS (Metal) available — $DEVICE_INFO"
        warn "MPS is slower than CUDA; inference may take several minutes"
        ;;
    cpu)
        warn "CPU only — inference will be very slow on large documents"
        ;;
    *)
        warn "Could not determine device"
        ;;
esac

# ── 5. .env file ──────────────────────────────────────────────────────────────
echo ""
echo "[ Configuration ]"

if [ -f ".env" ]; then
    pass ".env present"
elif [ -f ".env.example" ]; then
    warn ".env missing — using defaults (copy .env.example to .env to customise)"
else
    fail ".env.example missing — repository may be incomplete"
fi

# Load .env for model check
if [ -f ".env" ]; then
    set -o allexport; source .env; set +o allexport
fi
MODEL_DIR="${MODEL_DIR:-baidu/Unlimited-OCR}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
PORT="${PORT:-7860}"

pass "MODEL_DIR = $MODEL_DIR"
pass "OUTPUT_DIR = $OUTPUT_DIR"
pass "PORT = $PORT"

# ── 6. Directories ────────────────────────────────────────────────────────────
echo ""
echo "[ Directories ]"

for DIR in outputs log; do
    if [ -d "$DIR" ]; then
        pass "$DIR/ exists"
    else
        mkdir -p "$DIR"
        pass "$DIR/ created"
    fi
done

# ── 7. HuggingFace model access (network) ─────────────────────────────────────
echo ""
echo "[ HuggingFace Model Access ]"

HF_CHECK=$(python -c "
from huggingface_hub import repo_info
import os, sys
model = os.environ.get('MODEL_DIR', 'baidu/Unlimited-OCR')
token = os.environ.get('HF_TOKEN') or None
if model.startswith('/') or (len(model) > 1 and model[1] == ':'):
    print('local:' + model)
    sys.exit(0)
try:
    info = repo_info(model, token=token)
    print('ok:' + model)
except Exception as e:
    print('err:' + str(e))
" 2>/dev/null)

HF_STATUS="${HF_CHECK%%:*}"
HF_DETAIL="${HF_CHECK#*:}"

case "$HF_STATUS" in
    ok)    pass "HF model accessible: $HF_DETAIL" ;;
    local) pass "Using local model: $HF_DETAIL" ;;
    err)   warn "Cannot reach HF model ($HF_DETAIL) — check HF_TOKEN or network" ;;
    *)     warn "HF check inconclusive" ;;
esac

# ── 8. app.py present ─────────────────────────────────────────────────────────
echo ""
echo "[ Application Files ]"
for F in app.py scripts/setup.sh scripts/run_web.sh; do
    [ -f "$F" ] && pass "$F" || fail "$F missing"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYN}══════════════════════════════════════════${RST}"
echo -e "  ${GRN}✔ Passed${RST}: $PASS   ${YLW}⚠ Warnings${RST}: $WARN   ${RED}✘ Failed${RST}: $FAIL"
echo -e "${CYN}══════════════════════════════════════════${RST}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some checks failed. Fix the issues above, then re-run setup:${RST}"
    echo "  bash scripts/setup.sh"
    exit 1
else
    echo -e "${GRN}Environment looks good! Start the web UI with:${RST}"
    echo "  bash scripts/run_web.sh"
    exit 0
fi
