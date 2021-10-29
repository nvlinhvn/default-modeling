from Cython.Build import cythonize
from distutils.core import setup

setup(
    name='default_modeling',
    version='0.0.1',
    description="Default Probability Estimation Library",
    author="Linh, V. Nguyen",
    author_email="linhvietnguyen.ee@gmail.com",
    ext_modules=cythonize(["default_modeling/default_modeling/interface/*.pyx"]),
)
