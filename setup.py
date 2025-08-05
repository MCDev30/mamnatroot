from setuptools import setup, Extension
from Cython.Build import cythonize # type: ignore
import numpy
import os

solver_pyx = os.path.join("src", "mamnatroot", "solver.pyx")

extensions = [
    Extension(
        name="mamnatroot.solver",
        sources=[solver_pyx],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
    )
)
