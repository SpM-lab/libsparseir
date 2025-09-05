import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def run(self):
        # Build submodule
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call([
            'cmake',
            os.path.abspath('libsparseir'),
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath("src/pylibsparseir")}',
            f'-DCMAKE_BUILD_TYPE=Release',
        ], cwd=build_dir)

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
    ext_modules=[Extension('dummy', sources=[])],  # dummy Extension（build_extを有効にするため）
    zip_safe=False,
    package_data=package_data,
    exclude_package_data={
        '': ['__pycache__', '*.pyc'],
    },
)