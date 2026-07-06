"""
Unlimited-OCR — Gradio Web UI
==============================

Wraps the Unlimited-OCR Transformers inference path with a browser-based
interface that supports single-image and PDF uploads.

Usage
-----
    python app.py                      # uses defaults from .env / env vars
    PORT=7860 python app.py

Environment Variables (see .env.example)
-----------------------------------------
    MODEL_DIR   HF model ID or local path  (default: baidu/Unlimited-OCR)
    PORT        Gradio listen port         (default: 7860)
    OUTPUT_DIR  Where to save .md results  (default: ./outputs)
    CUDA_DEVICE CUDA_VISIBLE_DEVICES value (default: 0)
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import time
from pathlib import Path

# ── Load .env if python-dotenv is available ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can be set manually

# ── Configuration (no hard-coded paths) ─────────────────────────────────────
MODEL_DIR: str = os.environ.get("MODEL_DIR", "baidu/Unlimited-OCR")
PORT: int = int(os.environ.get("PORT", "7860"))
OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
CUDA_DEVICE: str = os.environ.get("CUDA_DEVICE", "0")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None

# Set CUDA device before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_DEVICE)

# ── Imports ──────────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402  (imported after env setup)
import torch  # noqa: E402

# ── Device selection ─────────────────────────────────────────────────────────
def _select_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _select_device()
DTYPE: torch.dtype = (
    torch.bfloat16 if DEVICE in ("cuda", "mps") else torch.float32
)

print(f"[Unlimited-OCR] Device: {DEVICE}  |  dtype: {DTYPE}")
print(f"[Unlimited-OCR] Model:  {MODEL_DIR}")
print(f"[Unlimited-OCR] Output: {OUTPUT_DIR}")

# ── Lazy model loader (load once on first request) ───────────────────────────
_model = None
_tokenizer = None


def _load_model():
    """Load model and tokenizer, caching them globally."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModel, AutoTokenizer

    print(f"[Unlimited-OCR] Loading tokenizer from {MODEL_DIR!r} …")
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    print(f"[Unlimited-OCR] Loading model from {MODEL_DIR!r} …")
    _model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=DTYPE,
        token=HF_TOKEN,
    )
    _model = _model.eval()

    if DEVICE == "cuda":
        _model = _model.cuda()
    elif DEVICE == "mps":
        _model = _model.to("mps")
    # cpu: leave as-is

    print("[Unlimited-OCR] Model ready.")
    return _model, _tokenizer


# ── PDF helper ───────────────────────────────────────────────────────────────
def _pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert a PDF to a list of PNG image paths (one per page)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


# ── Core inference functions ─────────────────────────────────────────────────
def _run_single_image(
    image_path: str,
    output_path: Path,
    config: str,
) -> str:
    """Run single-image OCR and return the markdown text."""
    model, tokenizer = _load_model()

    if config == "gundam":
        kwargs = dict(
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )
    else:  # base
        kwargs = dict(
            base_size=1024,
            image_size=1024,
            crop_mode=False,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_path,
        output_path=str(output_path),
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
        **kwargs,
    )

    # The model saves a .md file named after the image stem
    stem = Path(image_path).stem
    saved = output_path / f"{stem}.md"
    if saved.exists():
        return saved.read_text(encoding="utf-8")

    # Fallback: find any newly written .md
    mds = sorted(output_path.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if mds:
        return mds[-1].read_text(encoding="utf-8")
    return "(No output file found — check server logs)"


def _run_multi_image(
    image_paths: list[str],
    output_path: Path,
    label: str,
) -> str:
    """Run multi-page OCR (PDF or image list) and return markdown."""
    model, tokenizer = _load_model()
    output_path.mkdir(parents=True, exist_ok=True)

    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_paths,
        output_path=str(output_path),
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )

    # Find the most-recently written markdown
    mds = sorted(output_path.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if mds:
        return mds[-1].read_text(encoding="utf-8")
    return "(No output file found — check server logs)"


# ── Gradio handler ───────────────────────────────────────────────────────────
def ocr_handler(
    file,
    config: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """
    Gradio submit handler.

    Returns
    -------
    markdown_text : str   (rendered in Markdown component)
    raw_text      : str   (shown in Textbox)
    saved_path    : str   (info label)
    """
    if file is None:
        return "", "", "⚠️ Please upload a file first."

    file_path: str = file.name if hasattr(file, "name") else str(file)
    ext = Path(file_path).suffix.lower()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ext == ".pdf":
            progress(0, desc="Converting PDF to images…")
            images = _pdf_to_images(file_path)
            progress(0.3, desc=f"Running OCR on {len(images)} page(s)…")
            result = _run_multi_image(images, run_dir, label=ts)
        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
            progress(0.1, desc="Running single-image OCR…")
            result = _run_single_image(file_path, run_dir, config)
        else:
            return (
                "",
                "",
                f"❌ Unsupported file type: {ext}. "
                "Upload PNG/JPG/WEBP/BMP or PDF.",
            )
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return "", "", f"❌ Error: {exc}\n\n```\n{tb}\n```"

    saved_md = next(run_dir.glob("*.md"), None)
    saved_info = (
        f"✅ Saved to {saved_md}" if saved_md else "⚠️ Result not saved."
    )
    return result, result, saved_info


# ── UI Layout ────────────────────────────────────────────────────────────────
CSS = """
/* ── Global reset / font ── */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}

/* ── Header gradient ── */
#header {
    background: linear-gradient(135deg, #1e2a45 0%, #0d1b2a 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid #2a3a55;
    text-align: center;
}
#header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
#header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Card panels ── */
.panel-card {
    background: #161b27 !important;
    border: 1px solid #2a3a55 !important;
    border-radius: 14px !important;
    padding: 4px !important;
}

/* ── Upload zone ── */
.upload-zone {
    background: #1a2236 !important;
    border: 2px dashed #3b5278 !important;
    border-radius: 12px !important;
    min-height: 200px !important;
    transition: border-color 0.2s;
}
.upload-zone:hover {
    border-color: #60a5fa !important;
}

/* ── Submit button ── */
#submit-btn {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    border-radius: 10px !important;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
#submit-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

/* ── Status bar ── */
#status-bar {
    background: #1a2236 !important;
    border: 1px solid #2a3a55 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
}

/* ── Tab active ── */
.tab-nav button.selected {
    border-bottom: 2px solid #60a5fa !important;
    color: #60a5fa !important;
}

/* ── Device badge ── */
#device-badge {
    font-size: 0.8rem;
    color: #64748b;
    text-align: right;
    padding: 4px 8px;
}
"""

def _device_badge() -> str:
    icons = {"cuda": "🟢 CUDA", "mps": "🟡 Apple MPS", "cpu": "🔵 CPU"}
    return icons.get(DEVICE, DEVICE)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Unlimited OCR") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML(textwrap.dedent(f"""
            <div id="header">
              <h1>⚡ Unlimited OCR</h1>
              <p>One-shot long-horizon document parsing powered by Baidu's
              Unlimited-OCR model</p>
              <div id="device-badge">{_device_badge()} &nbsp;|&nbsp;
                Model: <code>{MODEL_DIR}</code></div>
            </div>
        """))

        # ── Main layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — inputs
            with gr.Column(scale=1, elem_classes="panel-card"):
                gr.Markdown("### 📄 Upload Document")
                file_input = gr.File(
                    label="Image or PDF",
                    file_types=[
                        ".png", ".jpg", ".jpeg",
                        ".webp", ".bmp", ".tiff", ".pdf",
                    ],
                    elem_classes="upload-zone",
                )
                config_radio = gr.Radio(
                    choices=["gundam", "base"],
                    value="gundam",
                    label="Single-image config",
                    info=(
                        "gundam: 640px + crop (recommended for dense docs). "
                        "base: 1024px no-crop. "
                        "Ignored for PDFs (always uses base)."
                    ),
                )
                submit_btn = gr.Button(
                    "🔍 Run OCR",
                    variant="primary",
                    elem_id="submit-btn",
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_id="status-bar",
                    lines=2,
                )

            # Right column — outputs
            with gr.Column(scale=2, elem_classes="panel-card"):
                gr.Markdown("### 📝 OCR Result")
                with gr.Tabs():
                    with gr.TabItem("📄 Markdown Preview"):
                        md_output = gr.Markdown(
                            value="*Output will appear here after running OCR*",
                            height=600,
                        )
                    with gr.TabItem("📋 Raw Text"):
                        raw_output = gr.Textbox(
                            label="Raw Markdown / Text",
                            lines=30,
                            interactive=False,
                        )

        # ── Examples ─────────────────────────────────────────────────────────
        sample_pdf = "Unlimited-OCR.pdf"
        if Path(sample_pdf).exists():
            gr.Markdown("### 💡 Quick Example")
            gr.Examples(
                examples=[[sample_pdf, "base"]],
                inputs=[file_input, config_radio],
                label="Run OCR on the bundled Unlimited-OCR.pdf",
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown(
            "<center><small>Results are saved to "
            f"<code>{OUTPUT_DIR}</code></small></center>",
        )

        # ── Wiring ───────────────────────────────────────────────────────────
        submit_btn.click(
            fn=ocr_handler,
            inputs=[file_input, config_radio],
            outputs=[md_output, raw_output, status_box],
        )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="violet",
            neutral_hue="slate",
        ),
        css=CSS,
    )
