import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        # Setup build environment by copying files from parent directory
        # Always run setup to ensure we have the latest source code
        print("Setting up build environment...")
        subprocess.check_call([sys.executable, 'setup_build.py'])
        
        # Build submodule
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)

        # Find the libsparseir source directory
        # Check multiple possible locations
        possible_paths = [
            '../',  # Development environment
            '.',    # Build environment
            '../libsparseir',
            'libsparseir',
            '..',   # Alternative build environment
            '../..' # Another possible location
        ]
        
        source_dir = None
        for path in possible_paths:
            cmake_file = os.path.join(path, 'CMakeLists.txt')
            if os.path.exists(cmake_file):
                source_dir = os.path.abspath(path)
                print(f"Found libsparseir source at: {source_dir}")
                break
        
        if source_dir is None:
            # Debug information
            print(f"Current working directory: {os.getcwd()}")
            print(f"Contents of current directory: {os.listdir('.')}")
            if os.path.exists('..'):
                print(f"Contents of parent directory: {os.listdir('..')}")
            raise RuntimeError("Could not find libsparseir source directory")

        # Configure CMake
        cmake_args = [
            'cmake',
            source_dir,
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath("src/pylibsparseir")}',
            f'-DCMAKE_BUILD_TYPE=Release',
        ]
        
        print(f"Running CMake with args: {cmake_args}")
        subprocess.check_call(cmake_args, cwd=build_dir)

        # Build
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_dir)

# Set package_data differently for sdist and wheel
if 'bdist_wheel' in sys.argv:
    package_data = {
        'pylibsparseir': ['libsparseir*.dylib', 'libsparseir*.so', 'libsparseir*.dll'],
    }
else:
    package_data = {}

setup(
    package_dir={'': 'src'},
    packages=['pylibsparseir'],
    cmdclass={'build_ext': CMakeBuild},
    ext_modules=[Extension('dummy', sources=[])],  # dummy Extension to enable build_ext
    zip_safe=False,
    package_data=package_data,
    exclude_package_data={
        '': ['__pycache__', '*.pyc'],
    },
)