import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

print("=" * 100, flush=True)
print("SETUP.PY EXECUTION STARTED!", flush=True)
print(f"Command line arguments: {sys.argv}", flush=True)
print(f"Current working directory: {os.getcwd()}", flush=True)
print(f"Directory contents: {os.listdir('.')}", flush=True)
print("=" * 100, flush=True)
sys.stdout.flush()

class CMakeBuild(build_ext):
    def run(self):
        print("\n" + "=" * 100, flush=True)
        print("CMAKE BUILD PROCESS STARTED - CMakeBuild.run() CALLED!", flush=True)
        print("=" * 100, flush=True)
        print(f"Python executable: {sys.executable}", flush=True)
        print(f"Current working directory: {os.getcwd()}", flush=True)
        print(f"build_ext extensions: {self.extensions}", flush=True)
        print(f"Extension names: {[ext.name for ext in self.extensions]}", flush=True)
        print("=" * 100, flush=True)
        sys.stdout.flush()

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

        # Find CMake executable
        cmake_exe = 'cmake'
        print("Searching for CMake executable...")
        # Try different cmake executables
        for cmake_candidate in ['cmake3', 'cmake']:
            try:
                print(f"Trying {cmake_candidate}...")
                result = subprocess.run([cmake_candidate, '--version'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cmake_exe = cmake_candidate
                    print(f"Found CMake: {cmake_exe}")
                    print(f"CMake version: {result.stdout.split()[2]}")
                    break
                else:
                    print(f"  {cmake_candidate} failed with exit code {result.returncode}")
                    print(f"  stdout: {result.stdout}")
                    print(f"  stderr: {result.stderr}")
                    continue
            except FileNotFoundError as e:
                print(f"  {cmake_candidate} not found: {e}")
                continue
        else:
            print("ERROR: No working CMake executable found!")
            print("Available executables in PATH:")
            try:
                result = subprocess.run(['which', 'cmake'], capture_output=True, text=True)
                print(f"  which cmake: {result.stdout.strip()}")
            except:
                pass
            try:
                result = subprocess.run(['which', 'cmake3'], capture_output=True, text=True)
                print(f"  which cmake3: {result.stdout.strip()}")
            except:
                pass

            raise RuntimeError("CMake not found. Please install CMake.")

        # Configure CMake
        output_dir_abs = os.path.abspath("pylibsparseir")
        print(f"Target output directory: {output_dir_abs}", flush=True)
        cmake_args = [
            cmake_exe,
            source_dir,
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir_abs}',
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir_abs}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={output_dir_abs}',
            f'-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_SHARED_LIBS=ON',
            '-DCMAKE_CXX_STANDARD=11',
            '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
            '-DSPARSEIR_USE_BLAS=ON',  # Enable BLAS support
        ]

        # Add architecture-specific flags for macOS
        import platform
        if platform.system() == 'Darwin':
            # Get the target architecture from environment or system
            target_arch = os.environ.get('ARCHFLAGS', '').replace('-arch ', '')
            if not target_arch:
                target_arch = platform.machine()

            cmake_args.extend([
                f'-DCMAKE_OSX_ARCHITECTURES={target_arch}',
                '-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0',
            ])

        print(f"Running CMake with args: {cmake_args}")
        print(f"CMake working directory: {build_dir}")
        print(f"CMake source directory: {source_dir}")
        try:
            result = subprocess.run(cmake_args, cwd=build_dir, capture_output=True, text=True, check=True)
            print(f"CMake configure stdout: {result.stdout}")
            print(f"CMake configure stderr: {result.stderr}")

            # Build
            print("Building with CMake...")
            build_result = subprocess.run([cmake_exe, '--build', '.', '--config', 'Release', '--verbose'],
                                        cwd=build_dir, capture_output=True, text=True, check=True)
            print(f"CMake build stdout: {build_result.stdout}", flush=True)
            print(f"CMake build stderr: {build_result.stderr}", flush=True)

            # Also try to build specifically the sparseir target
            print("Attempting to build sparseir target specifically...", flush=True)
            try:
                sparseir_result = subprocess.run([cmake_exe, '--build', '.', '--target', 'sparseir'],
                                                cwd=build_dir, capture_output=True, text=True, check=True)
                print(f"sparseir target build stdout: {sparseir_result.stdout}", flush=True)
                print(f"sparseir target build stderr: {sparseir_result.stderr}", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"sparseir target build failed: {e}", flush=True)
                print(f"sparseir target build stdout: {e.stdout}", flush=True)
                print(f"sparseir target build stderr: {e.stderr}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            try:
                if hasattr(e, 'stdout') and e.stdout:
                    print("----- CMake stdout (captured) -----")
                    print(e.stdout)
                if hasattr(e, 'stderr') and e.stderr:
                    print("----- CMake stderr (captured) -----")
                    print(e.stderr)
            except Exception:
                pass
            print("CMake build is required for this package!")
            raise RuntimeError(f"Failed to build native library: {e}")

        # Debug: Check what files were created
        output_dir = os.path.abspath("pylibsparseir")
        print(f"Checking output directory: {output_dir}")
        output_has_libs = False
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Files in output directory: {files}")

            # Check architecture of any library files and record presence
            for file in files:
                if file.endswith(('.dylib', '.so', '.dll')):
                    output_has_libs = True
                    lib_path = os.path.join(output_dir, file)
                    print(f"Checking library {file}:")
                    try:
                        result = subprocess.run(['file', lib_path], capture_output=True, text=True)
                        print(f"  file output: {result.stdout.strip()}")

                        if file.endswith('.dylib'):
                            result = subprocess.run(['lipo', '-info', lib_path], capture_output=True, text=True)
                            print(f"  lipo output: {result.stdout.strip()}")
                        elif file.endswith('.so'):
                            result = subprocess.run(['readelf', '-h', lib_path], capture_output=True, text=True)
                            print(f"  readelf output: {result.stdout}")
                    except Exception as e:
                        print(f"  Error checking {file}: {e}")
        else:
            print(f"Output directory does not exist!")

        # Always check build directory for any generated libraries
        print(f"Contents of build directory: {os.listdir(build_dir)}", flush=True)
        print("Searching for ALL files (especially libsparseir*) in build directory:", flush=True)

        found_libraries = []
        for root, dirs, files in os.walk(build_dir):
            # Only print directories that might contain libraries
            if any(f.endswith(('.so', '.dylib', '.dll', '.a')) or 'sparseir' in f for f in files):
                print(f"Directory: {root}", flush=True)
                print(f"  Files: {files}", flush=True)

            for file in files:
                # Look for any sparseir library files specifically
                if 'sparseir' in file and file.endswith(('.dylib', '.so', '.dll', '.a')):
                    lib_path = os.path.join(root, file)
                    found_libraries.append(lib_path)
                    print(f"FOUND SPARSEIR LIBRARY: {lib_path}", flush=True)

                    # Copy to output directory
                    dest_path = os.path.join(output_dir, file)
                    os.makedirs(output_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(lib_path, dest_path)
                    print(f"COPIED {lib_path} to {dest_path}", flush=True)

                    # Verify the copied file
                    if os.path.exists(dest_path):
                        size = os.path.getsize(dest_path)
                        print(f"Verified copy: {dest_path} ({size} bytes)", flush=True)

        if not found_libraries:
            print("ERROR: No libsparseir libraries found in build directory!", flush=True)
            print("Checking if CMake actually built the sparseir target...", flush=True)

            # Look for any file with 'sparseir' in the name
            sparseir_files = []
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if 'sparseir' in file.lower():
                        sparseir_files.append(os.path.join(root, file))

            if sparseir_files:
                print("Found files with 'sparseir' in name:", flush=True)
                for f in sparseir_files:
                    print(f"  {f}", flush=True)
            else:
                print("No files containing 'sparseir' found at all!", flush=True)

        # If we are building a wheel, ensure that at least one binary library is present.
        # Otherwise the wheel will be platform-tagged (e.g., linux_x86_64) without containing
        # any native code, which is invalid for PyPI/TestPyPI policies.
        if 'bdist_wheel' in sys.argv and not (found_libraries or output_has_libs):
            raise RuntimeError(
                "Wheel build aborted: no libsparseir binary libraries were produced by CMake. "
                "Ensure the native library is built and copied into 'pylibsparseir/'."
            )

        # Ensure binary libraries are present in build_lib so that wheel includes them.
        # build_py runs before build_ext, so copy the produced libraries here into build_lib.
        if 'bdist_wheel' in sys.argv:
            try:
                build_py_cmd = self.get_finalized_command('build_py')
                build_py_cmd.ensure_finalized()
                build_lib_dir = getattr(build_py_cmd, 'build_lib', None)
                if build_lib_dir:
                    dest_pkg_dir = os.path.join(build_lib_dir, 'pylibsparseir')
                    os.makedirs(dest_pkg_dir, exist_ok=True)
                    print(f"Copying binary libraries into wheel build dir: {dest_pkg_dir}")
                    try:
                        import glob
                        import shutil
                        lib_patterns = [
                            os.path.join(output_dir, 'libsparseir*.so*'),
                            os.path.join(output_dir, 'libsparseir*.dylib'),
                            os.path.join(output_dir, 'libsparseir*.dll'),
                        ]
                        copied_any = False
                        for pattern in lib_patterns:
                            for src_path in glob.glob(pattern):
                                if os.path.isfile(src_path):
                                    dst_path = os.path.join(dest_pkg_dir, os.path.basename(src_path))
                                    shutil.copy2(src_path, dst_path)
                                    print(f"  Copied {src_path} -> {dst_path}")
                                    copied_any = True
                        if not copied_any:
                            print("No binary libraries matched patterns for copying into build_lib")
                    except Exception as copy_err:
                        print(f"Error while copying libraries into build_lib: {copy_err}")
                else:
                    print("build_py build_lib path not available; cannot copy binary libraries into wheel build dir")
            except Exception as e:
                print(f"Failed to access build_py command for copying binaries: {e}")

        # Also check if CMake actually configured to build a shared library
        print("Checking CMake cache for library type:", flush=True)
        cmake_cache = os.path.join(build_dir, 'CMakeCache.txt')
        if os.path.exists(cmake_cache):
            with open(cmake_cache, 'r') as f:
                for line in f:
                    if 'BUILD_SHARED_LIBS' in line or 'LIBRARY_OUTPUT' in line or 'sparseir' in line.lower():
                        print(f"CMakeCache: {line.strip()}", flush=True)
        else:
            print("CMakeCache.txt not found!", flush=True)

        # After CMake build is complete, update package_data for wheel building
        if 'bdist_wheel' in sys.argv:
            self.get_package_data_after_build()

    def get_package_data_after_build(self):
        """Determine package_data after CMake build is complete"""
        print("Determining package_data after CMake build...", flush=True)

        import glob
        binary_files = []
        patterns = ['pylibsparseir/libsparseir*.dylib', 'pylibsparseir/libsparseir*.so*', 'pylibsparseir/libsparseir*.dll']
        for pattern in patterns:
            matches = glob.glob(pattern)
            binary_files.extend(matches)
            print(f"Pattern '{pattern}' matches: {matches}", flush=True)

        # Always include binary files if they exist in pylibsparseir directory
        print("Contents of pylibsparseir directory:", flush=True)
        if os.path.exists('pylibsparseir'):
            files = os.listdir('pylibsparseir')
            print(f"  {files}", flush=True)
            # Look for any library files
            lib_files = [f for f in files if f.startswith('libsparseir') and (f.endswith(('.so', '.dylib', '.dll')) or '.so.' in f)]
            if lib_files:
                binary_files.extend([f'pylibsparseir/{f}' for f in lib_files])
                print(f"Found library files in directory: {lib_files}", flush=True)

        if binary_files:
            package_data = {
                'pylibsparseir': ['libsparseir*', '*.so*', '*.dylib', '*.dll'],
            }
            print(f"Including binary files in wheel: {binary_files}", flush=True)
            print(f"Package data patterns: {package_data}", flush=True)

            # Update the distribution's package_data
            if hasattr(self, 'distribution') and self.distribution:
                self.distribution.package_data = package_data
                print("Updated distribution.package_data", flush=True)
        else:
            print("No binary files found after CMake build", flush=True)

        return binary_files

class CustomBuildPy(build_py):
    def run(self):
        # First run the normal build_py
        super().run()

        # If we're building a wheel and have built extensions, copy library files
        if 'bdist_wheel' in sys.argv and os.path.exists('pylibsparseir'):
            print("CustomBuildPy: Copying library files after build_ext...", flush=True)
            import glob
            import shutil

            lib_files = glob.glob('pylibsparseir/libsparseir*')
            if lib_files:
                print(f"Found library files to copy: {lib_files}", flush=True)
                # Copy to build directory
                for lib_file in lib_files:
                    if os.path.isfile(lib_file):
                        dest_dir = os.path.join(self.build_lib, 'pylibsparseir')
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, os.path.basename(lib_file))
                        shutil.copy2(lib_file, dest_path)
                        print(f"Copied {lib_file} to {dest_path}", flush=True)
            else:
                print("No library files found to copy", flush=True)

# Set initial package_data - include library files for wheels
print(f"Checking for binary files. sys.argv: {sys.argv}", flush=True)
if 'bdist_wheel' in sys.argv:
    print("Building wheel - including library files in package_data", flush=True)
    # Include library files that will be built by CMake
    package_data = {
        'pylibsparseir': ['libsparseir*', '*.so*', '*.dylib', '*.dll'],
    }
else:
    package_data = {}
    print("Not building wheel, no package_data needed", flush=True)

print("Setting up setuptools with CMakeBuild extension...", flush=True)

# Create a more explicit extension that forces build_ext to run
ext_modules = [
    Extension(
        'pylibsparseir._sparseir',  # More specific name
        sources=[],  # No sources needed, CMake handles compilation
        include_dirs=[],
        libraries=[],
        library_dirs=[],
        language='c++',
    )
]

print(f"Extension modules: {ext_modules}", flush=True)

setup(
    package_dir={'': './'},
    packages=['pylibsparseir'],
    cmdclass={'build_ext': CMakeBuild, 'build_py': CustomBuildPy},
    ext_modules=ext_modules,
    zip_safe=False,
    package_data=package_data,
    exclude_package_data={
        '': ['__pycache__', '*.pyc'],
    },
)