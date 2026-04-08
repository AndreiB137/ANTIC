from .kdv import run_kdv
from .kolmogorov import run_kolmogorov
try:
    from .bssn import run_bssn
except ImportError:
    run_bssn = None
