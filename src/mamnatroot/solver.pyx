# Fichier : mamnatroot/solver.pyx

# Imports Python
import numpy as np
import time
from typing import Callable, List, Tuple, Union
import inspect # Ajouté pour la fonctionnalité d'affichage

# Imports Cython pour la performance
cimport numpy as np
from libc.math cimport sqrt, fabs

# Initialisation de l'API C de NumPy
np.import_array()

# Bloc pour la visualisation
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# --- Fonctions C optimisées (au niveau du module) ---

# Fonction C pure pour la récursion
cdef list _c_isolate_recursive(object func, double a, double b, int current_depth, int max_depth):
    # La logique ici est une traduction exacte de votre fonction Python.
    
    # Cas de base : profondeur maximale atteinte
    if current_depth == max_depth:
        # On ne garde l'intervalle que si la fonction change de signe
        if func(a) * func(b) < 0:
            return [(a, b)]
        return []

    # Étape récursive
    cdef double mid = (a + b) / 2
    
    # Appel récursif pour les deux moitiés
    left_intervals = _c_isolate_recursive(func, a, mid, current_depth + 1, max_depth)
    right_intervals = _c_isolate_recursive(func, mid, b, current_depth + 1, max_depth)

    # Combinaison des résultats
    return left_intervals + right_intervals


# Wrapper accessible depuis Python qui lance la récursion C
cpdef list _isolate_roots_intervals(object func, double a, double b, int current_depth, int max_depth):
    return _c_isolate_recursive(func, a, b, current_depth, max_depth)


# Fonction C pour l'approximation
cdef double _approximate_root_in_interval(object func, tuple interval, bint verbose):
    # La logique ici est aussi une traduction exacte de votre fonction Python.
    # bint est le type Cython pour un booléen.
    cdef double c, d
    cdef double alpha, beta, gamma, discriminant, sqrt_discriminant, root1, root2
    c, d = interval

    x_points = np.array([c, (c + d) / 2, d])
    y_points = func(x_points)

    A = np.vstack([x_points**2, x_points, np.ones(3)]).T
    try:
        alpha, beta, gamma = np.linalg.solve(A, y_points)
    except np.linalg.LinAlgError:
        return (c + d) / 2

    if fabs(alpha) < 1e-12:
        return -gamma / beta if fabs(beta) > 1e-12 else (c + d) / 2

    discriminant = beta**2 - 4 * alpha * gamma

    if discriminant < 0:
        if verbose:
            print(f"    Warning: Negative discriminant. Simple approximation used.")
        return (c + d) / 2

    sqrt_discriminant = sqrt(discriminant)
    root1 = (-beta - sqrt_discriminant) / (2 * alpha)
    root2 = (-beta + sqrt_discriminant) / (2 * alpha)

    if c <= root1 <= d:
        return root1
    elif c <= root2 <= d:
        return root2
    else:
        if verbose:
            print(f"    Warning: Parabola root out of interval. Simple approximation used.")
        return (c + d) / 2


# --- La classe Python qui sert d'interface publique ---
class MamNatRootSolver:
    """
    Implements the hybrid MamNatRoot method for root finding.
    This Cython version is optimized for high performance.
    """
    @staticmethod
    def find_all_roots(
        func: Callable[[float], float],
        interval: Union[List[float], Tuple[float, float]],
        depth: int = 14,
        verbose: bool = False,
        visualize: bool = False,
        getRuntime: bool = False
    ) -> Union[List[float], Tuple[List[float], float]]:
        """
        Find all roots of a function in a given interval using the MamNatRoot method.
        """
        start_time = time.perf_counter()
        
        if len(interval) != 2 or interval[0] >= interval[1]:
            raise ValueError("Interval must be a list or tuple of two numbers [a, b] with a < b.")
        
        cdef double a, b
        a, b = interval

        if verbose:
            print("--- START MamNatRoot SOLVER ---")
            print(f"Interval: [{a}, {b}], Depth: {depth}")

        # Etape 1: Appel à la fonction Cython
        if verbose: print("\n1. Root isolation phase...")
        root_intervals = _isolate_roots_intervals(func, a, b, 0, depth)

        if not root_intervals:
            if verbose: print("No interval with sign change was found.")
            return ([], 0.0) if getRuntime else []

        if verbose: print(f"  {len(root_intervals)} interval(s) found.")

        # Etape 2: Appel à la fonction Cython
        if verbose: print("\n2. Local approximation phase...")
        found_roots = []
        for iv in root_intervals:
            if verbose: print(f"  Processing interval [{iv[0]:.4f}, {iv[1]:.4f}]...")
            root = _approximate_root_in_interval(func, iv, verbose)
            found_roots.append(root)
            if verbose: print(f"    -> Approximate root found at x = {root:.6f}")

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if verbose:
            print("\n--- END MamNatRoot SOLVER ---")
            print(f"Total computation time: {execution_time:.6f} seconds")

        # Etape 3: Visualisation
        if visualize:
            if not MATPLOTLIB_AVAILABLE:
                print("\nWarning: `matplotlib` is not installed. Visualization not possible.")
                print("Install it with: pip install mamnatroot[visualize]")
            else:
                if verbose: print("Generating plot...")
                x_vals = np.linspace(a, b, 100)
                y_vals = func(x_vals)
                plt.plot(x_vals, y_vals, label='f(x)', color='blue')
                plt.axhline(0, color='gray', linestyle='--')
                plt.plot(found_roots, func(np.array(found_roots)), 'rX', markersize=5, label='Zeros Found')
                plt.xlabel("x"); plt.ylabel("f(x)")
                plt.legend(); plt.grid(True); plt.show()

        # Etape 4: Retour
        if getRuntime:
            return (found_roots, execution_time)
        else:
            return found_roots