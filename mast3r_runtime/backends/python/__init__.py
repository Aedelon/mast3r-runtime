"""Pure Python fallback backend.

Uses numpy for all operations. Slow but works everywhere.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from .python_engine import PythonEngine

__all__ = ["PythonEngine"]
