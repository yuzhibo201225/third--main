#!/usr/bin/env python3
"""
Python 3.8 compatibility checker
Scans Python files for syntax that requires Python 3.9+
"""
import re
import sys
from pathlib import Path

def check_file(filepath):
    """Check a single file for Python 3.9+ syntax."""
    issues = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Skip comments
        code_part = line.split('#')[0]
        
        # Check for dict[...], list[...], set[...], tuple[...] in type hints
        if re.search(r':\s*(dict|list|set|tuple)\[', code_part):
            issues.append(f"Line {i}: lowercase generic type hint (use Dict/List/Set/Tuple from typing)")
        
        # Check for | None or | union syntax (but not in comments or strings)
        if re.search(r'\|\s*None|\w+\s*\|\s*\w+', code_part) and 'from __future__' not in code_part:
            # Exclude false positives like "auto|pt|onnx|trt" in string comments
            if not re.search(r'["\'].*\|.*["\']', code_part):
                issues.append(f"Line {i}: union operator | (use Optional or Union from typing)")
    
    return issues

def main():
    root = Path('campus_bike_detection')
    py_files = list(root.glob('*.py'))
    
    all_issues = {}
    for filepath in py_files:
        issues = check_file(filepath)
        if issues:
            all_issues[str(filepath)] = issues
    
    if all_issues:
        print("❌ Python 3.8 compatibility issues found:\n")
        for filepath, issues in all_issues.items():
            print(f"{filepath}:")
            for issue in issues:
                print(f"  - {issue}")
            print()
        sys.exit(1)
    else:
        print("✅ All files are Python 3.8 compatible!")
        sys.exit(0)

if __name__ == '__main__':
    main()
