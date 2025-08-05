# mamnatroot

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)

A smart, robust, and easy-to-use root-finding solver for Python. `mamnatroot` finds all roots of a function within a given interval, including hard-to-find roots that touch the x-axis without crossing it (roots of even multiplicity).

This project was born from an original idea by Mamlankou and Natabou, combining a recursive isolation phase with local quadratic approximation for maximum efficiency and precision.

## ✨ Key Features

*   **Comprehensive**: Finds all roots in an interval, not just the first one.
*   **Intelligent**: Detects both standard (crossing) roots and tangent roots (even multiplicity) without requiring the function's derivative.
*   **Simple API**: A single, clear `find_all_roots` function with straightforward options.
*   **Flexible**: Options to display calculation steps (`verbose`), visualize results (`visualize`), and measure execution time (`getRuntime`).
*   **Robust**: Based on a hybrid approach that guarantees not to miss solutions while precisely refining them.

## 💾 Installation

`mamnatroot` only requires `numpy` for its core functionality. `matplotlib` is an optional dependency needed for visualization.

**Prerequisites:** Python 3.8 or higher.

You can install the package using `pip`.

#### Basic Installation (without visualization)

Open a terminal or command prompt and run:
```bash
pip install mamnatroot
```
*(This command will work once the package is published to PyPI. For now, you are using it locally with Poetry.)*


---

### OS-Specific Instructions

Installation via `pip` is universal. Here is how to open a terminal on major operating systems:

*   **Windows**: Open **PowerShell** or **Command Prompt (CMD)** from the Start Menu.
*   **macOS**: Open the **Terminal** app (found in `Applications/Utilities`).
*   **Linux**: Open your favorite terminal emulator (e.g., Gnome Terminal, Konsole, xterm).

Once the terminal is open, the `pip install ...` commands above will work the same way.

## 🚀 Quick Start

Using `mamnatroot` is extremely simple. Here is a basic example to find the roots of `f(x) = x² - 4`.

```python
import numpy as np
from mamnatroot import MamNatRootSolver

# 1. Define your function
def f(x):
    return x**2 - 4

# 2. Call the solver on the interval [-5, 5]
roots = MamNatRootSolver.find_all_roots(f, interval=[-5, 5])

# 3. Print the results
print(f"Roots found: {roots}")
# Expected output: Roots found: [-2.0, 2.0]
```

## 💡 Advanced Example: Detecting Tangent Roots

The real power of `mamnatroot` is its ability to find roots of even multiplicity. Let's take the example of `f(x) = (x + 3)(x - 1)²`, which has a standard root at `x = -3` and a tangent root at `x = 1`.

```python
import numpy as np
from mamnatroot import MamNatRootSolver

# 1. Define the function with a tangent root
def f(x):
    return (x + 3) * (x - 1)**2

# 2. Use the solver with all options enabled
#    - verbose=True: to see what the algorithm is doing
#    - visualize=True: to display the plot
#    - getRuntime=True: to get the execution time
result = MamNatRootSolver.find_all_roots(
    func=f,
    interval=[-5, 5],
    depth=18,  # A higher depth helps to accurately pinpoint tangent roots
    verbose=True,
    visualize=True,
    getRuntime=True
)

# 3. Process the result (which is a tuple)
roots, exec_time = result

print("\n--- FINAL RESULT ---")
print(f"Roots found: {np.round(roots, 6)}")
print(f"Execution time: {exec_time:.6f} seconds")
```

#### Expected output (example):
```
--- START MamNatRoot Solver ---
Interval: [-5, 5], Depth: 18

1. Isolating roots (crossing and tangent)...
  ... (search details)
  65 candidate interval(s) found.

2. Local approximation phase...

--- END MamNatRoot Solver ---
Final roots found: [-3.0000000000000004, 1.0000000000000002]
Total computation time: 0.002451 seconds

--- FINAL RESULT ---
Roots found: [-3. 1.]
Execution time: 0.002451 seconds
```
*(A plot will also be displayed, showing the two points found on the function's curve.)*

## 🛠️ API Reference: `find_all_roots`

```python
MamNatRootSolver.find_all_roots(func, interval, depth=14, verbose=False, visualize=False, getRuntime=False)
```

| Parameter    | Type                                   | Description                                                                                             | Default   |
|--------------|----------------------------------------|---------------------------------------------------------------------------------------------------------|-----------|
| `func`       | `Callable[[float], float]`             | The single-argument function whose roots are to be found.                                               | Required  |
| `interval`   | `List[float]` or `Tuple[float, float]` | The search interval `[a, b]`.                                                                           | Required  |
| `depth`      | `int`                                  | The depth of the recursive subdivision. Increase for better precision on tangent roots.                 | `14`      |
| `verbose`    | `bool`                                 | If `True`, prints the details of the search process.                                                    | `False`   |
| `visualize`  | `bool`                                 | If `True`, displays a plot of the results (requires `matplotlib`).                                      | `False`   |
| `getRuntime` | `bool`                                 | If `True`, the function returns a `(roots, exec_time)` tuple. Otherwise, it returns only the list of roots. | `False`   |

## 🧠 How It Works

`mamnatroot` uses a two-phase hybrid method:

1.  **Smart Isolation Phase**: The solver recursively subdivides the interval. Unlike classic methods that only keep subintervals where the function's sign changes, `mamnatroot` also keeps those where the function "dips" towards the x-axis without crossing. This allows it to "see" areas where tangent roots might exist.

2.  **Quadratic Approximation Phase**: For each candidate interval found, the solver fits a parabola through three points on the function. It then calculates the root of this parabola, which provides a very precise approximation of the function's true root in that interval.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find a bug, feel free to open an issue or submit a pull request on the project's GitHub repository.

## ✍️ Authors

*   **Charbel Mamlankou** 
*   **Jean-Eudes Natabou**
## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.