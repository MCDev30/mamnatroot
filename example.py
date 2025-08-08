import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from mamnatroot import MamNatRootSolver

def f(x):
    return np.sin(x**2) - 0.5


roots, time = MamNatRootSolver.find_all_roots(f, interval=[-4, 2], getRuntime=True, verbose=True, visualize=True, depth=8)

print(f"\nSol = {np.round(roots, 5)}")
