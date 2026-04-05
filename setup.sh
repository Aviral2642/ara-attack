#!/usr/bin/env bash
# One-command setup for ARA attack project.
# Usage: bash setup.sh [--cpu | --cuda | --mps]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BACKEND="${1:---auto}"

# --- 1. Python version check -------------------------------------------------
REQUIRED_MIN="3.10"
REQUIRED_MAX="3.13"   # 3.14 upstream wheels not yet available for torch 2.4

PY_BIN="${PYTHON:-python3}"
PY_VER=$("$PY_BIN" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if ! "$PY_BIN" -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)"; then
  echo "[ERROR] Python $PY_VER detected. Need >=3.10,<3.14. Use pyenv:"
  echo "  pyenv install 3.11.10 && pyenv local 3.11.10"
  exit 1
fi
echo "[ok] Python $PY_VER"

# --- 2. venv -----------------------------------------------------------------
if [[ ! -d .venv ]]; then
  "$PY_BIN" -m venv .venv
  echo "[ok] created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools >/dev/null

# --- 3. torch backend --------------------------------------------------------
case "$BACKEND" in
  --cuda)
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 ;;
  --cpu)
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu ;;
  --mps|--auto)
    # macOS default wheel has MPS; Linux default is CPU
    pip install torch==2.4.1 ;;
esac

# --- 4. remaining deps -------------------------------------------------------
pip install -r requirements.txt

# --- 5. HF token --------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN not set. LLaMA-2/3 and Gemma are gated models."
  echo "       Request access at https://huggingface.co/meta-llama and export HF_TOKEN=hf_..."
fi

# --- 6. results symlinks for paper figures ------------------------------------
ln -sfn "$REPO_ROOT/results/figures" "$REPO_ROOT/paper/ccs/figures"
ln -sfn "$REPO_ROOT/results/figures" "$REPO_ROOT/paper/neurips/figures"

# --- 7. sanity test ----------------------------------------------------------
python -c "import torch, transformers; print(f'torch {torch.__version__}  transformers {transformers.__version__}')"
python -c "import torch; print('cuda:', torch.cuda.is_available(), ' mps:', torch.backends.mps.is_available())"

echo "[ok] setup complete. Activate with: source .venv/bin/activate"
