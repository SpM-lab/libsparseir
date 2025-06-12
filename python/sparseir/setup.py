#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


class BuildCommand(build_py):
    def run(self):
        # Build the C library
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Copy Makefile.bundle to the package directory
        makefile_src = os.path.join(root_dir, 'bundle', 'Makefile.bundle')
        makefile_dst = os.path.join(os.path.dirname(__file__), 'Makefile.bundle')
        shutil.copy2(makefile_src, makefile_dst)
        
        # Build using the copied Makefile
        subprocess.check_call(['make', '-f', 'Makefile.bundle'], 
                            cwd=os.path.dirname(__file__))
        
        # Create lib directory if it doesn't exist
        lib_dir = os.path.join(os.path.dirname(__file__), 'lib')
        os.makedirs(lib_dir, exist_ok=True)
        
        # Copy the built library
        if os.name == 'nt':  # Windows
            lib_name = 'libsparseir.dll'
        elif os.name == 'posix':  # Linux/Unix
            lib_name = 'libsparseir.so'
        else:  # macOS
            lib_name = 'libsparseir.dylib'
        
        subprocess.check_call(['cp', os.path.join(os.path.dirname(__file__), 
                                                lib_name), lib_dir])
        
        # Run the original build_py
        build_py.run(self)


class SDistCommand(sdist):
    def run(self):
        # Copy source files and headers
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pkg_dir = os.path.dirname(__file__)
        
        # Create libsparseir directory
        libsparseir_dir = os.path.join(pkg_dir, 'libsparseir')
        if os.path.exists(libsparseir_dir):
            shutil.rmtree(libsparseir_dir)
        os.makedirs(libsparseir_dir)
        
        # Copy src directory
        src_dir = os.path.join(libsparseir_dir, 'src')
        shutil.copytree(os.path.join(root_dir, 'src'), src_dir)
        
        # Copy include directory
        include_dir = os.path.join(libsparseir_dir, 'include', 'sparseir')
        os.makedirs(os.path.dirname(include_dir), exist_ok=True)
        shutil.copytree(os.path.join(root_dir, 'include', 'sparseir'), include_dir)
        
        # Remove duplicate directories if they exist
        for dir_name in ['include', 'src']:
            dir_path = os.path.join(pkg_dir, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
        # Run the original sdist
        sdist.run(self)


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
    cmdclass=dict(
        build_py=BuildCommand,
        sdist=SDistCommand,
    ),
    package_data={
        'sparseir': [
            'lib/*.so', 'lib/*.dylib', 'lib/*.dll',
            'Makefile.bundle',
            'libsparseir/src/*.cpp',
            'libsparseir/include/sparseir/*.h',
        ],
    },
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