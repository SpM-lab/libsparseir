#!/usr/bin/env python3
import os
import re
from setuptools import setup, find_packages


def get_version():
    version_h = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'include', 'sparseir', 'version.h'
    )
    with open(version_h, 'r') as f:
        content = f.read()
    
    major = re.search(r'#define SPARSEIR_VERSION_MAJOR (\d+)', content).group(1)
    minor = re.search(r'#define SPARSEIR_VERSION_MINOR (\d+)', content).group(1)
    patch = re.search(r'#define SPARSEIR_VERSION_PATCH (\d+)', content).group(1)
    
    return f"{major}.{minor}.{patch}"


setup(
    name="sparseir",
    version=get_version(),
    description="Python bindings for libsparseir",
    author="Hiroshi Shinaoka",
    author_email="h.shinaoka@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics"
    ],
) 