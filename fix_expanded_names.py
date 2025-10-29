#!/usr/bin/env python3
"""
Fix the expanded Fortran code by replacing remaining NAME macros with actual function names.
"""

import re

def fix_expanded_names(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all function definitions and their names
    function_pattern = r'SUBROUTINE (evaluate_tau_\w+_\d+d)'
    functions = re.findall(function_pattern, content)
    
    # Process each function individually
    lines = content.split('\n')
    result_lines = []
    current_function = None
    
    for line in lines:
        # Check if this line starts a new function
        match = re.match(r'SUBROUTINE (evaluate_tau_\w+_\d+d)', line)
        if match:
            current_function = match.group(1)
            result_lines.append(line)
        else:
            # If we're inside a function and see errore('NAME', replace it
            if current_function and "errore('NAME'" in line:
                line = line.replace("errore('NAME'", f"errore('{current_function}'")
            result_lines.append(line)
    
    # Join the lines back together
    content = '\n'.join(result_lines)
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed {len(functions)} functions in {output_file}")

if __name__ == "__main__":
    fix_expanded_names("sparseir_ext_expanded.F90", "sparseir_ext_expanded_fixed2.F90")
