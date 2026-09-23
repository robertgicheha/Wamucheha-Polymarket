"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure the repo root is importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
