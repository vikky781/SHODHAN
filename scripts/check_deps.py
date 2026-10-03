#!/usr/bin/env python3
"""Dependency guard for SHODHAN (pure Python 3, standard library only).

Fails when any of the following reference a forbidden third-party solver or
numerical library:

  1. CMakeLists.txt and other CMake files (comments are ignored),
  2. #include directives in src/ and include/,
  3. the shared libraries linked by built binaries (via ldd or otool -L when
     one of them is available),
  4. test-only code (tests/support/) leaking into the library: src/ and include/
     must not include it, the CMake definitions of shodhan_core and of the CLI
     must not mention tests/, and the built library must not contain test objects.

Usage:
  python scripts/check_deps.py [--build-dir DIR] [--verbose]

--verbose prints every file and binary that was checked.
Exit status: 0 clean, 1 forbidden reference found.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

# Names matched as whole alphanumeric tokens (case-insensitive).
FORBIDDEN_EXACT = {
    "highs", "scip", "clp", "cbc", "coin", "glpk", "amd", "eigen", "blas",
    "lapack", "cblas", "lapacke", "openblas", "cudss", "cusolver",
    # SuiteSparse components that are easy to pull in under their own names.
    "klu", "csparse", "cxsparse", "colamd", "ccolamd", "camd",
}
# Names matched as token prefixes (case-insensitive), e.g. cusolverDn.h.
FORBIDDEN_PREFIX = (
    "coinor", "ortools", "suitesparse", "cholmod", "umfpack", "cusolver",
    "cudss", "lapack", "openblas", "cblas",
)

# Names that may carry a version suffix (Eigen3, blas64, ...). Deliberately not
# amd/clp/cbc/coin: "AMD64" is a processor name, not the AMD ordering library.
FORBIDDEN_VERSIONED = {"eigen", "highs", "scip", "glpk", "blas", "lapack", "cblas", "openblas"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "build", "out", "data", "__pycache__", ".venv", "venv"}


def forbidden_in(text):
    """Returns the forbidden names found in `text` (tokenized)."""
    found = set()
    for token in re.split(r"[^A-Za-z0-9]+", text):
        if not token:
            continue
        t = token.lower()
        candidates = {t}
        if t.startswith("lib") and len(t) > 3:
            candidates.add(t[3:])  # libblas -> blas
        for c in list(candidates):
            unversioned = c.rstrip("0123456789")
            if unversioned in FORBIDDEN_VERSIONED:
                candidates.add(unversioned)
        for c in candidates:
            if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIX):
                found.add(c)
    return found


def walk(*subdirs):
    for sub in subdirs:
        base = os.path.join(ROOT, sub)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("build")]
            for name in filenames:
                yield os.path.join(dirpath, name)


def cmake_files():
    for name in os.listdir(ROOT):
        if name == "CMakeLists.txt" or name.endswith(".cmake"):
            yield os.path.join(ROOT, name)
    for path in walk("cmake", "src", "include", "tests"):
        base = os.path.basename(path)
        if base == "CMakeLists.txt" or base.endswith(".cmake"):
            yield path
    presets = os.path.join(ROOT, "CMakePresets.json")
    if os.path.exists(presets):
        yield presets


def strip_cmake_comments(line):
    # Good enough for this project: '#' starts a comment outside of quotes.
    out = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        if ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out)


def check_cmake(verbose, problems):
    for path in sorted(set(cmake_files())):
        rel = os.path.relpath(path, ROOT)
        if verbose:
            print("checking CMake file: " + rel)
        is_json = path.endswith(".json")
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                text = line if is_json else strip_cmake_comments(line)
                for name in forbidden_in(text):
                    problems.append("%s:%d: references forbidden name '%s'" % (rel, lineno, name))


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def check_includes(verbose, problems):
    exts = (".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx", ".cu", ".cuh")
    for path in sorted(walk("src", "include")):
        if not path.endswith(exts):
            continue
        rel = os.path.relpath(path, ROOT)
        if verbose:
            print("checking includes: " + rel)
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                m = INCLUDE_RE.match(line)
                if m:
                    for name in forbidden_in(m.group(1)):
                        problems.append(
                            "%s:%d: #include of forbidden library '%s' (%s)"
                            % (rel, lineno, name, m.group(1)))


SUPPORT_RE = re.compile(r"support/|dense_ref_lp|dense_lu|lu_testing|work_check|random_lp|test_harness|test_models")


def cmake_call_text(text, start_pattern):
    """Text of the first CMake call that starts with start_pattern (up to its closing paren)."""
    m = re.search(start_pattern, text)
    if not m:
        return ""
    depth = 0
    for i in range(m.start(), len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1]
    return text[m.start():]


def check_test_support_isolation(build_dirs, verbose, problems):
    # (a) library and CLI sources must not include test-only headers.
    exts = (".h", ".hpp", ".hh", ".hxx", ".cpp", ".cc", ".cxx")
    for path in sorted(walk("src", "include")):
        if not path.endswith(exts):
            continue
        rel = os.path.relpath(path, ROOT)
        if verbose:
            print("checking test-support isolation: " + rel)
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                m = INCLUDE_RE.match(line)
                if m and SUPPORT_RE.search(m.group(1)):
                    problems.append("%s:%d: library code includes test-only code (%s)"
                                    % (rel, lineno, m.group(1)))

    # (b) the CMake definitions of the library and the CLI must not mention tests/.
    cmake = os.path.join(ROOT, "CMakeLists.txt")
    if os.path.exists(cmake):
        with open(cmake, encoding="utf-8", errors="replace") as f:
            text = "\n".join(strip_cmake_comments(line) for line in f.read().splitlines())
        if verbose:
            print("checking CMake test-support isolation: CMakeLists.txt")
        definitions = {
            "add_library(shodhan_core": cmake_call_text(text, r"add_library\s*\(\s*shodhan_core"),
            "add_executable(shodhan": cmake_call_text(text, r"add_executable\s*\(\s*shodhan\s"),
        }
        # Every set()/file(GLOB)/list() that builds the library or CLI source lists.
        for m in re.finditer(r"(set|list|file)\s*\(", text):
            call = cmake_call_text(text[m.start():], r"(set|list|file)\s*\(")
            if re.search(r"SHODHAN_(OPTIONAL_)?CORE_SOURCES|SHODHAN_CLI_SOURCES", call):
                definitions["source list: " + call.split()[0] + " " + call.split()[1][:40]] = call
        for what, body in definitions.items():
            if "tests" in body:
                problems.append("CMakeLists.txt: the definition of %s mentions tests/ "
                                "(test-only code must not reach the library or the CLI)" % what)

    # (c) the built library must not contain test objects.
    ar = shutil.which("ar")
    if ar is None:
        return
    for build_dir in build_dirs:
        for dirpath, dirnames, filenames in os.walk(build_dir):
            dirnames[:] = [d for d in dirnames if d != "CMakeFiles"]
            for name in filenames:
                if name in ("libshodhan_core.a", "shodhan_core.lib"):
                    path = os.path.join(dirpath, name)
                    try:
                        res = subprocess.run([ar, "t", path], capture_output=True, text=True, timeout=60)
                    except (OSError, subprocess.SubprocessError):
                        continue
                    if verbose:
                        print("checking library members: " + os.path.relpath(path, ROOT))
                    for member in res.stdout.split():
                        if SUPPORT_RE.search(member) or member.startswith("test_"):
                            problems.append("%s contains test-only object %s"
                                            % (os.path.relpath(path, ROOT), member))


def find_binaries(build_dir):
    binaries = []
    for dirpath, dirnames, filenames in os.walk(build_dir):
        dirnames[:] = [d for d in dirnames if d != "CMakeFiles"]
        for name in filenames:
            base = name.lower()
            if base in ("shodhan", "shodhan.exe", "shodhan_tests", "shodhan_tests.exe"):
                binaries.append(os.path.join(dirpath, name))
    return sorted(binaries)


def check_binaries(build_dirs, verbose, problems):
    tool = None
    if shutil.which("ldd"):
        tool = ["ldd"]
    elif shutil.which("otool"):
        tool = ["otool", "-L"]
    if tool is None:
        print("note: neither ldd nor otool found; skipping linked-library check")
        return
    checked = 0
    for build_dir in build_dirs:
        for binary in find_binaries(build_dir):
            rel = os.path.relpath(binary, ROOT)
            try:
                res = subprocess.run(tool + [binary], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as e:
                print("note: could not inspect %s: %s" % (rel, e))
                continue
            checked += 1
            if verbose:
                print("checking linked libraries: " + rel)
            for line in (res.stdout + res.stderr).splitlines():
                for name in forbidden_in(line):
                    problems.append("%s: links forbidden library '%s' (%s)" % (rel, name, line.strip()))
    if checked == 0:
        print("note: no built binaries found; linked-library check skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", action="append", default=None,
                    help="build directory to inspect (repeatable); default: every build*/ directory")
    ap.add_argument("--verbose", action="store_true", help="print everything that was checked")
    args = ap.parse_args()

    build_dirs = args.build_dir
    if build_dirs is None:
        build_dirs = []
        for top in sorted(os.listdir(ROOT)):
            if top == "build" or top.startswith("build-"):
                build_dirs.append(os.path.join(ROOT, top))

    problems = []
    check_cmake(args.verbose, problems)
    check_includes(args.verbose, problems)
    check_binaries(build_dirs, args.verbose, problems)
    check_test_support_isolation(build_dirs, args.verbose, problems)

    if problems:
        print("dependency check FAILED:")
        for p in problems:
            print("  " + p)
        return 1
    print("dependency check passed: no forbidden library referenced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
