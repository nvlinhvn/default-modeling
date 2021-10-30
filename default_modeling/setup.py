from distutils.core import setup
import setuptools
from Cython.Build import cythonize

setup(
    name='default_modeling',
    version='0.0.1',
    description="Default Probability Estimation Library",
    author="Linh, V. Nguyen",
    author_email="linhvietnguyen.ee@gmail.com",
    url="https://github.com/nvlinhvn/default-modeling/default_modeling",
    include_package_data=True,
    install_requires=[
        "pandas>=1.3.4",
        "numpy>=1.21.3",
        "scikit-learn>=1.0.0",
        "category_encoders>=2.3.0",
        "Cython>=0.29.21",
        "scipy>=1.7.0",
    ],
    extras_require={"dev": ["joblib" ]},
    ext_modules=cythonize(["default_modeling/default_modeling/utils/*.pyx",
                           "default_modeling/default_modeling/interface/*.pyx"],
                          language_level = "3"),
)
