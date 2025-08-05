import numpy as np
import time
from typing import Callable, List, Tuple, Union

cimport numpy as np
from libc.math cimport sqrt, fabs

np.import_array()

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


cdef list _c_isolate_crossing_recursive(object func, double a, double b, int current_depth, int max_depth):
    if current_depth == max_depth:
        if func(a) * func(b) < 0:
            return [(a, b)]
        return []
    cdef double mid = (a + b) / 2
    if np.isclose(a, mid): 
        if func(a) * func(b) < 0: return [(a, b)]
        return []
    left = _c_isolate_crossing_recursive(func, a, mid, current_depth + 1, max_depth)
    right = _c_isolate_crossing_recursive(func, mid, b, current_depth + 1, max_depth)
    return left + right


# Moteur N°2: Approximation dans un intervalle (inchangé)
cdef double _approximate_root_in_interval(object func, tuple interval):
    cdef double c, d, alpha, beta, gamma, discriminant, sqrt_discriminant, root1, root2
    c, d = interval
    x_points = np.array([c, (c+d)/2, d])
    y_points = func(x_points)
    # Ensure y_points is a 1D array of length 3
    if hasattr(y_points, "__len__") and len(y_points) == 3:
        y_points = np.asarray(y_points, dtype=np.float64)
    else:
        # If func does not support vectorized input, fallback to scalar calls
        y_points = np.array([func(c), func((c+d)/2), func(d)], dtype=np.float64)
    A = np.vstack([x_points**2, x_points, np.ones(3)]).T
    try:
        sol = np.linalg.solve(A, y_points)
        alpha, beta, gamma = sol[0], sol[1], sol[2]
    except np.linalg.LinAlgError:
        return (c+d)/2
    if fabs(alpha) < 1e-12:
        return -gamma / beta if fabs(beta) > 1e-12 else (c+d)/2
    discriminant = beta**2 - 4*alpha*gamma
    if discriminant < 0:
        return (c+d)/2
    sqrt_discriminant = sqrt(discriminant)
    root1 = (-beta - sqrt_discriminant)/(2*alpha)
    root2 = (-beta + sqrt_discriminant)/(2*alpha)
    if c <= root1 <= d:
        return root1
    elif c <= root2 <= d:
        return root2
    else:
        return (c+d)/2


# --- CLASSE PYTHON (INTERFACE PUBLIQUE) ---
class MamNatRootSolver:
    @staticmethod
    def find_all_roots(
        func: Callable[[float], float],
        interval: Union[List[float], Tuple[float, float]],
        depth: int = 10,
        verbose: bool = False,
        visualize: bool = False,
        getRuntime: bool = False
    ) -> Union[List[float], Tuple[List[float], float]]:
        
        start_time = time.perf_counter()
        
        if len(interval) != 2 or interval[0] >= interval[1]:
            raise ValueError("Interval must be a list or tuple [a, b] with a < b.")
        
        cdef double a, b, tol, validation_tol, h
        a, b = interval
        tol = 1e-12
        validation_tol = 1e-8
        h = 1e-8 # Pas pour la différence finie

        # Fonction interne pour approximer la dérivée numériquement
        def f_prime_approx(x):
            x = np.asarray(x)
            return (func(x + h) - func(x - h)) / (2 * h)

        if verbose:
            print("--- START MamNatRoot SOLVER ---")
            print(f"Interval: [{a}, {b}]")

        # Etape 0: Vérification des bornes
        initial_roots = []
        if fabs(func(a)) < tol: initial_roots.append(a)
        if fabs(func(b)) < tol and not (len(initial_roots) > 0 and np.isclose(b, initial_roots[0])):
            initial_roots.append(b)
        # if verbose and initial_roots: print(f"Found root(s) on interval boundaries: {initial_roots}")
        
        # Etape 1: Recherche des racines croissantes (sur f)
        # if verbose: print("\n1a. Searching for crossing roots on f(x)...")
        crossing_intervals = _c_isolate_crossing_recursive(func, a, b, 0, depth)
        # if verbose: print(f"  -> Found {len(crossing_intervals)} interval(s) with sign changes.")
        
        crossing_roots = []
        for iv in crossing_intervals:
            root = _approximate_root_in_interval(func, iv)
            crossing_roots.append(root)

        # Etape 2: Recherche des racines tangentes
        # if verbose: print("\n1b. Searching for tangent root candidates on f'(x)...")
        tangent_intervals = _c_isolate_crossing_recursive(f_prime_approx, a, b, 0, depth)
        # if verbose: print(f"  -> Found {len(tangent_intervals)} potential tangent location(s).")
        
        tangent_candidates = []
        for iv in tangent_intervals:
            root = _approximate_root_in_interval(f_prime_approx, iv)
            tangent_candidates.append(root)

        # Etape 3: Filtrage, combinaison et dé-duplication
        # if verbose: print("\n2. Filtering and combining all results...")
        
        tangent_roots = [r for r in tangent_candidates if fabs(func(r)) < validation_tol]
        
        all_found_roots = sorted(list(set(initial_roots + crossing_roots + tangent_roots)))
        
        found_roots = []
        if all_found_roots:
            found_roots = [all_found_roots[0]]
            for root in all_found_roots[1:]:
                if not np.isclose(root, found_roots[-1], atol=1e-8):
                    found_roots.append(root)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if verbose:
            print("\n--- END MamNatRoot SOLVER ---")
            # print(f"Final roots found: {found_roots}")
            print(f"Total computation time: {execution_time:.6f} seconds")

        if visualize:
            if not MATPLOTLIB_AVAILABLE: print("\nWarning: `matplotlib` is not installed.")
            else:
                if verbose:
                    print("Generating plot...")
                x_vals = np.linspace(a, b, 200)
                y_vals = func(x_vals)
                plt.plot(x_vals, y_vals, label='f(x)', color='blue')
                plt.axhline(0, color='gray', linestyle='--')
                
                if found_roots:
                    plt.plot(found_roots, func(np.array(found_roots)), 'rX', markersize=5, label='Roots', linestyle='None')
                    for root in found_roots:
                        plt.axvline(x=root, color='black', linestyle=':', alpha=0.6)
                        plt.text(root+0.15, 0.1 * np.max(y_vals), f"x={root:.5f}", rotation=90, va='bottom', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()
        if getRuntime:
            return (found_roots, execution_time)
        else:
            return found_roots