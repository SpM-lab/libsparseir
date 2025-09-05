#!/usr/bin/env python3
"""
Setup script to prepare the build environment by copying necessary files
from the parent libsparseir directory to the current directory.
"""

import os
import shutil
import sys

def setup_build_environment():
    """Copy necessary files from parent directory to current directory for build."""
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    # Files and directories to copy
    items_to_copy = [
        'CMakeLists.txt',
        'LICENSE',
        'include',
        'src',
        'fortran',
        'cmake'
    ]
    
    print(f"Setting up build environment in: {current_dir}")
    print(f"Copying from parent directory: {parent_dir}")
    
    for item in items_to_copy:
        src_path = os.path.join(parent_dir, item)
        dst_path = os.path.join(current_dir, item)
        
        if os.path.exists(src_path):
            # Always remove existing and copy fresh
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
                print(f"Copied directory: {item}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"Copied file: {item}")
        else:
            print(f"Warning: {item} not found in parent directory")

if __name__ == "__main__":
    setup_build_environment()
