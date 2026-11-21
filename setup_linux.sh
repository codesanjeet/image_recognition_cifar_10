#!/usr/bin/env bash
# One-time setup: creates a Python virtual environment and installs everything.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then
  echo "python3 not found. Install it first, e.g.:"
  echo "  Ubuntu/Debian: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  echo "  Fedora:        sudo dnf install -y python3 python3-pip"
  echo "  Arch:          sudo pacman -S python python-pip"
  exit 1
fi

python3 -m venv cifar_env || { echo "Run: sudo apt install python3-venv  (then re-run this script)"; exit 1; }
source cifar_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Setup done. Now run:  source cifar_env/bin/activate && python predict.py bird.jpg"
