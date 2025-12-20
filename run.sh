#!/usr/bin/env bash
set -euo pipefail

# Git Bash
# ./run.sh
if [ -f "venv/Scripts/activate" ]; then
  # shellcheck source=/dev/null
  . "venv/Scripts/activate"
elif [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  . "venv/bin/activate"
else
  echo "Warning: venv activate script not found (expected venv/Scripts/activate or venv/bin/activate)"
fi

python app_tk.py