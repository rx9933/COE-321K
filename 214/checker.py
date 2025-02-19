import numpy as np
import os
path = "../testingtruss/2dtestcase/"
file1 = path+"truesolution.txt"
file2 = path+"_multiinputsolution.txt"


loc = path
filenames = [loc+"truesolution"]
for filename in filenames:

    # Check if the file exists without an extension
    if os.path.exists(filename) and not os.path.exists(filename + ".txt"):
        os.rename(filename, filename + ".txt")
        print(f'Renamed "{filename}" to "nodes.txt"')
    else:
        print(f'"{filename}" does not exist or "nodes.txt" already exists.')




def is_float(value):
    """Check if a value can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def compare_files(file1, file2, atol=1e-6, rtol=1e-5):
    """Compare two text files line by line, checking floats with np.isclose."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    max_lines = max(len(lines1), len(lines2))
    for i in range(max_lines):
        line1 = lines1[i].strip() if i < len(lines1) else "[MISSING LINE]"
        line2 = lines2[i].strip() if i < len(lines2) else "[MISSING LINE]"
        
        if line1 == line2:
            continue  # Lines are identical
        
        tokens1 = line1.split()
        tokens2 = line2.split()
        
        if len(tokens1) != len(tokens2):
            print(f"Line {i+1} differs in length:")
            print(f"File1: {line1}")
            print(f"File2: {line2}")
            print("---")
            continue
        
        differences = []
        for t1, t2 in zip(tokens1, tokens2):
            if is_float(t1) and is_float(t2):
                if not np.isclose(float(t1), float(t2), atol=atol, rtol=rtol):
                    differences.append((t1, t2))
            elif t1 != t2:
                differences.append((t1, t2))
        
        if differences:
            print(f"Line {i+1} differs:")
            print(f"File1: {line1}")
            print(f"File2: {line2}")
            print("Differences:")
            for d1, d2 in differences:
                print(f"  {d1}  !=  {d2}")
            print("---")
compare_files(file1, file2)