
from distutils.core import setup
import setuptools

setup(
    name='default_modeling',
    version='0.0.1',
    description="Default Probability Estimation Library",
    author="Linh, V. Nguyen",
    author_email="linhvietnguyen.ee@gmail.com",
    packages=['default_modeling', 'default_modeling.interface', 'default_modeling.utils'],
    include_package_data=True,
    install_requires=[
        "pandas>=0.25.0",
        "numpy=>1.17.1",
        "scikit-learn>=0.23.0",
        "scipy>=0.18.1",
        "category_encoders>=0.23.0"
    ]
)
