"""Vercel entry point; the package lives under src/, so put it on the import path first."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nl_sql.serving import create_app

app = create_app()
