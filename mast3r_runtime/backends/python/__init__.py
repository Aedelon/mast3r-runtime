"""Pure Python fallback backend.

Uses numpy for all operations. Slow but works everywhere.

Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
"""

from .dune_model import DUNEModel, load_dune_model
from .python_engine import PythonEngine

__all__ = ["PythonEngine", "DUNEModel", "load_dune_model"]
