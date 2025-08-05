import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from mamnatroot import MamNatRootSolver

def f(x):
    return (x + 3)*(x - 1)**2


roots, time = MamNatRootSolver.find_all_roots(f, interval=[-4, 2], getRuntime=True, verbose=True, visualize=True, depth=10)

print(f"\nRésultat final: Racines trouvées = {np.round(roots, 5)}")
