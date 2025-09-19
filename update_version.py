#!/usr/bin/env python3
"""
Version Update Script for libsparseir

Updates version numbers across C++ and Python components:
- include/sparseir/version.h (C++ library)
- python/pyproject.toml (Python package)

Usage:
    python update_version.py 0.4.3
    python update_version.py 1.0.0
"""

import sys
import re
import os
from pathlib import Path

def update_cpp_version(version_parts, repo_root):
    """Update C++ version in include/sparseir/version.h"""
    version_h_path = repo_root / "include" / "sparseir" / "version.h"
    
    if not version_h_path.exists():
        print(f"Error: {version_h_path} not found")
        return False
    
    # Read current content
    with open(version_h_path, 'r') as f:
        content = f.read()
    
    # Update version macros
    content = re.sub(
        r'#define SPARSEIR_VERSION_MAJOR \d+',
        f'#define SPARSEIR_VERSION_MAJOR {version_parts[0]}',
        content
    )
    content = re.sub(
        r'#define SPARSEIR_VERSION_MINOR \d+',
        f'#define SPARSEIR_VERSION_MINOR {version_parts[1]}',
        content
    )
    content = re.sub(
        r'#define SPARSEIR_VERSION_PATCH \d+',
        f'#define SPARSEIR_VERSION_PATCH {version_parts[2]}',
        content
    )
    
    # Write updated content
    with open(version_h_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated C++ version in {version_h_path}")
    return True

def update_python_version(version_string, repo_root):
    """Update Python version in python/pyproject.toml"""
    pyproject_path = repo_root / "python" / "pyproject.toml"
    
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        return False
    
    # Read current content
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Update version in [project] section only (not cmake.version)
    content = re.sub(
        r'(\[project\][^\[]*?)version = "[^"]*"',
        f'\\1version = "{version_string}"',
        content,
        flags=re.DOTALL
    )
    
    # If no version field exists, add it after name
    if 'version = ' not in content:
        content = re.sub(
            r'(name = "[^"]*")',
            f'\\1\nversion = "{version_string}"',
            content
        )
    
    # Write updated content
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated Python version in {pyproject_path}")
    return True


def validate_version(version_string):
    """Validate version string format (x.y.z)"""
    pattern = r'^\d+\.\d+\.\d+$'
    if not re.match(pattern, version_string):
        print(f"Error: Invalid version format '{version_string}'. Expected format: x.y.z (e.g., 0.4.3)")
        return False
    return True

def parse_version(version_string):
    """Parse version string into components"""
    parts = version_string.split('.')
    return [int(part) for part in parts]

def get_cpp_version(repo_root):
    """Get current C++ version from version.h"""
    version_h_path = repo_root / "include" / "sparseir" / "version.h"
    if not version_h_path.exists():
        return None
    
    with open(version_h_path, 'r') as f:
        content = f.read()
    
    major_match = re.search(r'#define SPARSEIR_VERSION_MAJOR (\d+)', content)
    minor_match = re.search(r'#define SPARSEIR_VERSION_MINOR (\d+)', content)
    patch_match = re.search(r'#define SPARSEIR_VERSION_PATCH (\d+)', content)
    
    if major_match and minor_match and patch_match:
        return f"{major_match.group(1)}.{minor_match.group(1)}.{patch_match.group(1)}"
    return None

def get_python_version(repo_root):
    """Get current Python version from pyproject.toml"""
    pyproject_path = repo_root / "python" / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    version_match = re.search(r'version = "([^"]*)"', content)
    if version_match:
        return version_match.group(1)
    return None

def show_current_versions(repo_root):
    """Show current versions in both files"""
    print("Current versions:")
    
    cpp_version = get_cpp_version(repo_root)
    python_version = get_python_version(repo_root)
    
    if cpp_version:
        print(f"  C++ (version.h): {cpp_version}")
    else:
        print(f"  C++ (version.h): Unable to parse or file not found")
    
    if python_version:
        print(f"  Python (pyproject.toml): {python_version}")
    else:
        print(f"  Python (pyproject.toml): No version field found or file not found")
    
    return cpp_version, python_version

def check_version_consistency(repo_root):
    """Check if versions are consistent across files"""
    cpp_version, python_version = show_current_versions(repo_root)
    print()
    
    if not cpp_version:
        print("❌ Could not read C++ version from include/sparseir/version.h")
        return False
    
    if not python_version:
        print("❌ Could not read Python version from python/pyproject.toml")
        return False
    
    if cpp_version == python_version:
        print(f"✅ Versions are consistent: {cpp_version}")
        return True
    else:
        print(f"❌ Version mismatch:")
        print(f"   C++: {cpp_version}")
        print(f"   Python: {python_version}")
        print()
        print(f"To fix this, run: python update_version.py {cpp_version}")
        return False

def main():
    repo_root = Path(__file__).parent
    
    if len(sys.argv) == 1:
        # No arguments - check version consistency
        print("Version Consistency Check")
        print("=" * 30)
        print()
        
        if check_version_consistency(repo_root):
            print()
            print("\U0001F44D All versions are in sync!")
        else:
            print()
            print("\U0001F6A8 Version inconsistency detected.")
            sys.exit(1)
        return
    
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python update_version.py           # Check version consistency")
        print("  python update_version.py <version> # Update all versions")
        print()
        print("Examples:")
        print("  python update_version.py           # Check current versions")
        print("  python update_version.py 0.4.3     # Update to version 0.4.3")
        print()
        
        # Show current versions
        show_current_versions(repo_root)
        sys.exit(1)
    
    version_string = sys.argv[1]
    repo_root = Path(__file__).parent
    
    # Validate version format
    if not validate_version(version_string):
        sys.exit(1)
    
    # Parse version components
    version_parts = parse_version(version_string)
    
    print(f"Updating version to {version_string}")
    print()
    
    # Show current versions
    show_current_versions(repo_root)
    print()
    
    # Update versions
    success = True
    success &= update_cpp_version(version_parts, repo_root)
    success &= update_python_version(version_string, repo_root)
    
    if success:
        print()
        print(f"✅ Successfully updated all versions to {version_string}")
        print()
        print("Updated files:")
        print("  - include/sparseir/version.h (C++ library)")
        print("  - python/pyproject.toml (Python package)")
        print("  - Conda recipe will automatically use version.h")
        print()
        print("Next steps:")
        print("1. Review the changes: git diff")
        print("2. Test the build: cd python && pip wheel .")
        print("3. Commit the changes: git add -A && git commit -m 'Bump version to {}'".format(version_string))
    else:
        print()
        print("❌ Some updates failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
