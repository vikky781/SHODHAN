#!/usr/bin/env python3
"""Expands .mps.gz files with the Python standard library, for machines where SHODHAN was built without zlib.

Usage:
  python scripts/gunzip_mps.py FILE.mps.gz [OUT.mps]      expand one file (default output: FILE.mps next to it)
  python scripts/gunzip_mps.py DIRECTORY                  expand every .mps.gz in the directory

An existing output file is kept unless --force is given. The gzip stream is read to its end, so a truncated or
corrupted file is reported instead of leaving a partial .mps behind.
Exit status: 0 when every file was expanded or already present, 1 otherwise.
"""
import argparse
import gzip
import os
import shutil
import sys


def expand(src, dst, force):
    if os.path.exists(dst) and not force:
        print("kept existing %s (use --force to overwrite)" % dst)
        return True
    tmp = dst + ".part"
    try:
        with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    except (OSError, EOFError) as e:  # gzip.BadGzipFile is an OSError
        print("cannot expand %s: %s" % (src, e))
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dst)
    print("%s -> %s (%d bytes)" % (src, dst, os.path.getsize(dst)))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a .mps.gz file or a directory")
    ap.add_argument("output", nargs="?", help="output file (single-file mode only)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args(argv)

    if os.path.isdir(args.path):
        if args.output:
            print("an output name is only valid for a single file")
            return 1
        files = sorted(f for f in os.listdir(args.path) if f.lower().endswith(".gz"))
        if not files:
            print("no .gz files in " + args.path)
            return 1
        ok = True
        for f in files:
            src = os.path.join(args.path, f)
            ok = expand(src, src[:-3], args.force) and ok
        return 0 if ok else 1
    if not os.path.isfile(args.path):
        print("not found: " + args.path)
        return 1
    dst = args.output or (args.path[:-3] if args.path.lower().endswith(".gz") else args.path + ".out")
    return 0 if expand(args.path, dst, args.force) else 1


if __name__ == "__main__":
    sys.exit(main())
