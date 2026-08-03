"""Add src/ to sys.path so tests can import polyllm without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
