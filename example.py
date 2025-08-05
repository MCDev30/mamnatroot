import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from mamnatroot import MamNatRootSolver

def f(x):
    return x**x - 49#(x + 3) * (x - 1)


roots, time = MamNatRootSolver.find_all_roots(
    f,
    interval=[0, 4],
    getRuntime=True,
    # depth=20,  
    # verbose=True,
    visualize=True
)

print(f"\nRésultat final: Racines trouvées = {np.round(roots, 10)}", time)
