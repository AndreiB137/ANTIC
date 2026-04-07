from .solver import Solver, State
from .kdv import KDVSolver
from .kolmogorov import KolmogorovSolver

# BSSNSolver requires JAX_NR — import lazily to avoid hard dependency.
try:
    from .bssn import BSSNSolver
except ImportError:
    BSSNSolver = None