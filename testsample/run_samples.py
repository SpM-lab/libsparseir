#!/usr/bin/env python3

import re
import sys
import os
import subprocess
from pathlib import Path


def extract_c_code_blocks(markdown_text):
    """Extract C code blocks from markdown text."""
    pattern = r'```c\n(.*?)```'
    return re.findall(pattern, markdown_text, re.DOTALL)


def generate_c_file(code_block, index):
    """Generate a C file from a code block."""
    # Extract includes and other code
    includes = []
    other_code = []
    
    for line in code_block.split('\n'):
        if line.strip().startswith('#include'):
            includes.append(line)
        else:
            other_code.append(line)
    
    # Generate C code
    c_code = ""
    
    # Add extracted includes
    if includes:
        c_code += '\n'.join(includes) + '\n'
    
    c_code += """
int main() {
"""
    
    # Add other code
    other_code = '\n'.join(other_code)
    # Remove main function if present
    other_code = re.sub(r'int\s+main\s*\([^)]*\)\s*{', '', other_code)
    other_code = re.sub(r'return\s+0;', '', other_code)
    other_code = re.sub(r'}\s*$', '', other_code)
    
    c_code += other_code
    c_code += "\n    return 0;\n}\n"
    return c_code


def compile_and_run_test(test_file):
    """Compile and run a test file."""
    # Get compiler from environment variable
    compiler = os.environ.get('CC', 'cc')
    
    # Set paths
    include_dir = Path("../include")
    build_dir = Path("../build")
    
    # Compile
    compile_cmd = [
        compiler,
        "-o",
        str(test_file.with_suffix('')),
        "-g",
        str(test_file),
        f"-I{include_dir}",
        f"-L{build_dir}",
        f"-Wl,-rpath,{build_dir}",  # Add rpath for macOS
        "-lsparseir"
    ]
    
    print(f"Compile command: {' '.join(compile_cmd)}")
    print(f"Compiling {test_file}...")
    result = subprocess.run(compile_cmd)
    if result.returncode != 0:
        print(f"Compilation failed for {test_file}")
        return False
    
    # Run
    exe_path = str(test_file.with_suffix(''))
    print(f"Running {exe_path}...")
    result = subprocess.run([exe_path])
    if result.returncode != 0:
        print(f"Test failed for {test_file}")
        return False
    
    return True


def main():
    # Set paths
    readme_path = Path("../README.md")
    build_dir = Path("../build")

    print("Debug information:")
    print(f"  README path: {readme_path}")
    print(f"  Build dir: {build_dir}")
    print(f"  Compiler: {os.environ.get('CC', 'cc')}")

    if not readme_path.exists():
        print(f"Error: {readme_path} not found")
        sys.exit(1)
    
    if not build_dir.exists():
        print(f"Error: {build_dir} not found")
        sys.exit(1)
    
    # Create output directory for test files
    output_dir = Path("test_files")
    output_dir.mkdir(exist_ok=True)
    
    with open(readme_path, "r") as f:
        readme_text = f.read()
    
    code_blocks = extract_c_code_blocks(readme_text)
    if not code_blocks:
        print("No C code blocks found in README.md")
        sys.exit(0)
    
    for i, block in enumerate(code_blocks, 1):
        test_code = generate_c_file(block, i)
        test_file = output_dir / f"readme_sample{i}.c"
        
        # Write test file
        with open(test_file, "w") as f:
            f.write(test_code)
        print(f"Generated test file: {test_file}")
        
        # Compile and run test
        if not compile_and_run_test(test_file):
            print(f"Test failed for {test_file}. Stopping execution.")
            sys.exit(1)


if __name__ == "__main__":
    main() 