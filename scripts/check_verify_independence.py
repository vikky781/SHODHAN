#!/usr/bin/env python3
"""Checks that the verifier in verify/ is independent of the solver and of third-party code.

Fails (exit 1) if a Python file under verify/
  * imports a module that is neither in the standard library nor one of the verifier's own packages, or
  * mentions the C++ sources (a path under src/ or include/, or a .cpp / .hpp file name).

The Python tests may RUN the solver as an executable (that produces inputs); they must not share code with it.

Usage: python scripts/check_verify_independence.py [--verify-dir verify]
"""

import argparse
import ast
import os
import re
import sys

OWN_PACKAGES = {"kasauti", "tests"}

# Used only when sys.stdlib_module_names (Python 3.10+) is not available.
FALLBACK_STDLIB = {
    "__future__", "argparse", "ast", "base64", "binascii", "bisect", "collections", "contextlib", "copy", "csv",
    "datetime", "decimal", "difflib", "enum", "errno", "fractions", "functools", "gzip", "hashlib", "heapq", "io",
    "itertools", "json", "math", "operator", "os", "pathlib", "random", "re", "shutil", "string", "struct", "subprocess",
    "sys", "tempfile", "textwrap", "time", "traceback", "typing", "unittest", "warnings", "zlib",
}

SOURCE_REFERENCE = re.compile(r"(\bsrc/|\binclude/|\.cpp\b|\.hpp\b)")


def stdlib_names():
    return set(getattr(sys, "stdlib_module_names", FALLBACK_STDLIB)) | ({"__future__"})


def python_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "build", "dist") and not d.endswith(".egg-info")]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(base, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default=os.path.join(os.path.dirname(__file__), "..", "verify"))
    a = ap.parse_args()
    root = os.path.normpath(a.verify_dir)
    if not os.path.isdir(root):
        print("verify directory not found: %s" % root)
        return 2
    stdlib = stdlib_names()
    problems = []
    checked = 0
    for path in python_files(root):
        checked += 1
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rel = os.path.relpath(path, root)
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError as e:
            problems.append("%s: cannot parse: %s" % (rel, e))
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in stdlib and name not in OWN_PACKAGES:
                    problems.append("%s:%d: imports %r, which is not in the standard library" % (rel, node.lineno, name))
        for no, line in enumerate(text.splitlines(), start=1):
            if SOURCE_REFERENCE.search(line):
                problems.append("%s:%d: refers to the C++ sources: %s" % (rel, no, line.strip()[:100]))
    for p in problems:
        print("FAIL " + p)
    print("verify independence: %d Python files checked, %d problem(s)" % (checked, len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
