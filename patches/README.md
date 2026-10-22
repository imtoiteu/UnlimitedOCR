# Upstream model compatibility patches

## Baidu Unlimited-OCR CPU inference

- Upstream repository: https://huggingface.co/baidu/Unlimited-OCR
- Upstream revision: `8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3`
- Patch: `baidu-unlimited-ocr-cpu.patch`

The patch replaces CUDA-only tensor placement with device-aware placement,
uses float32 image tensors when CUDA is unavailable, and disables CUDA
autocast when CUDA is unavailable.

The patch was captured from a working CPU-inference experiment on a CPU-only
Ubuntu VPS. It has not yet been validated against a clean model download or
on CUDA/MPS hardware.

This patch is a review artifact; it is not automatically applied by the
application or setup script.
