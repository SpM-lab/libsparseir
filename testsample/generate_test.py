import re
import sys
from pathlib import Path


def extract_c_code_blocks(markdown_text):
    """Extract C/C++ code blocks from markdown text."""
    pattern = r'```(?:c|cpp)\n(.*?)```'
    return re.findall(pattern, markdown_text, re.DOTALL)


def generate_test_file(code_blocks):
    """Generate a test file from code blocks."""
    test_code = """#include <sparseir/sparseir.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

TEST_CASE("README code samples", "[readme]") {
"""
    
    for i, block in enumerate(code_blocks):
        # Remove main function if present
        block = re.sub(r'int\s+main\s*\([^)]*\)\s*{', '', block)
        block = re.sub(r'return\s+0;', '', block)
        block = re.sub(r'}\s*$', '', block)
        
        test_code += f"""
    SECTION("Sample {i+1}") {{
{block}
    }}
"""
    
    test_code += "}\n"
    return test_code


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_test.py <README_PATH> <TEST_FILE>")
        sys.exit(1)

    readme_path = Path(sys.argv[1])
    test_file = Path(sys.argv[2])

    if not readme_path.exists():
        print(f"Error: {readme_path} not found")
        sys.exit(1)
    
    with open(readme_path, "r") as f:
        readme_text = f.read()
    
    code_blocks = extract_c_code_blocks(readme_text)
    if not code_blocks:
        print("No C/C++ code blocks found in README.md")
        sys.exit(0)
    
    test_code = generate_test_file(code_blocks)
    with open(test_file, "w") as f:
        f.write(test_code)
    
    print(f"Generated test file: {test_file}")


if __name__ == "__main__":
    main() 