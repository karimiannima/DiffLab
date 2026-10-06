#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin git@github.com:karimiannima/DiffLab.git
fi
git branch -M main
git push -u origin main
echo "Done. Enable GitHub Pages: Settings → Pages → branch main / root"
