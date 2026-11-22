#!/usr/bin/env bash
# run.sh — one command to set up (if needed) and launch the Insurance RAG web UI.
# Safe to re-run: it only does setup work that is actually missing.
set -euo pipefail

# Always work from the folder this script lives in, so it runs from anywhere.
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/Scripts/python.exe"   # Windows virtualenv layout

echo "==> Insurance RAG — starting up"

# 1) Create the virtualenv the first time only.
if [ ! -f "$PY" ]; then
  echo "==> Creating virtual environment (.venv) ..."
  py -3 -m venv "$VENV"
fi

# 2) Install dependencies only if they're not already importable (fast re-runs).
if "$PY" -c "import streamlit, insurance_rag" >/dev/null 2>&1; then
  echo "==> Dependencies already installed."
else
  echo "==> Installing dependencies (first time only) ..."
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  "$PY" -m pip install --quiet -e ".[ui]"
fi

# 3) Make sure Ollama has the 'mistral' model (the app needs it to answer).
if command -v ollama >/dev/null 2>&1; then
  if ! ollama list 2>/dev/null | grep -qi "mistral"; then
    echo "==> Pulling the 'mistral' model (first time only, ~4 GB) ..."
    ollama pull mistral
  fi
else
  echo "!! Ollama not found on PATH — install & launch it from https://ollama.com"
  echo "   (the UI will still open, but answers need Ollama running)."
fi

# 4) Launch the web UI. First ingest downloads the embedding model (~670 MB) once.
echo "==> Opening the app at http://localhost:8501   (press Ctrl+C here to stop)"
exec "$PY" -m streamlit run app.py
