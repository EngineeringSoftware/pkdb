"""
Expression evaluation system for pkdb.

This module provides expression evaluation capabilities for debugging PyKokkos code,
allowing users to evaluate arbitrary Python-like expressions in both Python (PDB)
and C++ (GDB/CUDA-GDB/HIP) debugging contexts.
"""

from .expression_evaluator import ExpressionEvaluator, EvaluationResult
from .context import EvaluationContext, ContextType

__all__ = [
    "ExpressionEvaluator",
    "EvaluationResult",
    "EvaluationContext",
    "ContextType",
]
