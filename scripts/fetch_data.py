#!/usr/bin/env python
"""Fetches benchmark instances into data/raw/ (standard library only; the files are never committed).

Sources (all downloads are recorded with URL, date and SHA-256 in data/MANIFEST.txt, which is git-ignored):

  netlib   LP test problems, https://www.netlib.org/lp/data/ . Most files use netlib's compressed MPS format; the
           expander source emps.c from the same directory is compiled with the C compiler found on PATH (cc, gcc or
           clang) and run on each file. Output: data/raw/netlib/NAME.mps
  maros    Maros-Meszaros convex QP set in QPS format, the files QPDATA1.ZIP, QPDATA2.ZIP and QPDATA3.ZIP on the author's
           page http://www.doc.ic.ac.uk/~im/ (plain http: the files cannot be authenticated by TLS, so their
           contents are additionally compared with the QPS copies in the public repository YimingYAN/QP-Test-Problems
           when --verify-maros is given). Output: data/raw/maros_meszaros/NAME.QPS
  miplib   MIPLIB 2017 benchmark set, https://miplib.zib.de/downloads/benchmark.zip . Output: data/raw/miplib/*.mps.gz

TLS is always verified. If the Python certificate store cannot verify a host (miplib.zib.de sends an incomplete
chain), give a CA bundle with --cafile PATH or the SSL_CERT_FILE environment variable; verification is never switched
off. A source that cannot be fetched or verified is reported and skipped.

Usage: python scripts/fetch_data.py [--only netlib,maros,miplib] [--cafile PATH] [--verify-maros] [--data-dir data]
The script is idempotent: files that already exist are kept (their hashes are still recorded).
"""
import argparse
import datetime
import hashlib
import io
import os
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile

UA = {"User-Agent": "Mozilla/5.0 (shodhan fetch_data.py)"}
NETLIB = "https://www.netlib.org/lp/data/"
MAROS = ["http://www.doc.ic.ac.uk/~im/QPDATA%d.ZIP" % i for i in (1, 2, 3)]
MAROS_MIRROR_API = "https://api.github.com/repos/YimingYAN/QP-Test-Problems/contents/QPS_Files"
MIPLIB = "https://miplib.zib.de/downloads/benchmark.zip"
# Files of the netlib directory that are not LP models.
NETLIB_SKIP = {"ascii", "changes", "emps.c", "emps.exe.gz", "emps.f", "mpc.src", "nams.ps.gz", "readme", "minos",
               "vtp.base", "stocfor3.old", "pilot.ja", "pilot.we", "maros", "maros-r7", "standata", "standgub", "standmps"}


class Fetcher:
    def __init__(self, cafile):
        ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
        self.opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    def get(self, url, timeout=120):
        req = urllib.request.Request(url, headers=UA)
        with self.opener.open(req, timeout=timeout) as r:
            return r.read()

    def to_file(self, url, path, timeout=600):
        req = urllib.request.Request(url, headers=UA)
        with self.opener.open(req, timeout=timeout) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(path + ".part", path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 4:
                        self.rows[parts[0]] = parts

    def add(self, relpath, url, sha, date=None):
        old = self.rows.get(relpath)
        self.rows[relpath] = [relpath, url, old[2] if old and old[3] == sha else (date or today()), sha]
        # columns: path, source url, download date (kept while the hash is unchanged), sha256

    def save(self):
        with open(self.path, "w") as f:
            f.write("# path\tsource url\tdownload date (UTC)\tsha256 (git-ignored; written by scripts/fetch_data.py)\n")
            for k in sorted(self.rows):
                r = self.rows[k]
                f.write("%s\t%s\t%s\t%s\n" % (r[0], r[1], r[2], r[3]))


def today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def build_emps(f, workdir):
    src = os.path.join(workdir, "emps.c")
    if not os.path.exists(src):
        f.to_file(NETLIB + "emps.c", src)
    exe = os.path.join(workdir, "emps" + (".exe" if os.name == "nt" else ""))
    if os.path.exists(exe):
        return exe
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            r = subprocess.run([cc, "-O1", "-w", "-o", exe, src], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(exe):
                return exe
            print("  %s could not compile emps.c: %s" % (cc, r.stderr.strip()[:200]))
    return None


def fetch_netlib(f, data, man):
    out = os.path.join(data, "raw", "netlib")
    work = os.path.join(out, "_download")
    os.makedirs(work, exist_ok=True)
    try:
        listing = f.get(NETLIB).decode("latin1")
    except Exception as e:
        print("netlib: cannot list %s: %s" % (NETLIB, e))
        return 0
    names = [n for n in re.findall(r'href="([^"?/][^"]*)"', listing) if "/" not in n and " " not in n and n not in NETLIB_SKIP]
    emps = build_emps(f, work)
    if emps is None:
        print("netlib: no C compiler could build the expander emps.c; the compressed files stay unexpanded")
    got = 0
    for n in names:
        raw = os.path.join(work, n)
        target = os.path.join(out, n + ".mps")
        try:
            if not os.path.exists(raw):
                f.to_file(NETLIB + n, raw)
        except Exception as e:
            print("netlib: %s failed: %s" % (n, e))
            continue
        man.add("netlib/_download/" + n, NETLIB + n, sha256_file(raw))
        if not os.path.exists(target):
            with open(raw, "rb") as fh:
                head = fh.read(8)
            if head.startswith(b"NAME"):
                shutil.copyfile(raw, target)
            elif head.startswith(b"# to unb"):
                print("netlib: %s is a shell archive (several parts), not expanded" % n)
                continue
            elif emps is not None:
                with open(raw, "rb") as fin, open(target + ".part", "wb") as fout:
                    r = subprocess.run([emps], stdin=fin, stdout=fout, stderr=subprocess.PIPE)
                if r.returncode != 0 or os.path.getsize(target + ".part") == 0:
                    print("netlib: expanding %s failed (%s)" % (n, r.stderr.decode(errors="replace")[:100]))
                    os.remove(target + ".part")
                    continue
                os.replace(target + ".part", target)
            else:
                continue
        man.add("netlib/" + n + ".mps", NETLIB + n + " (expanded with emps.c from the same directory)", sha256_file(target))
        got += 1
    print("netlib: %d models in %s" % (got, out))
    return got


def fetch_maros(f, data, man, verify):
    out = os.path.join(data, "raw", "maros_meszaros")
    work = os.path.join(out, "_download")
    os.makedirs(work, exist_ok=True)
    count = 0
    for url in MAROS:
        z = os.path.join(work, url.rsplit("/", 1)[1])
        try:
            if not os.path.exists(z):
                f.to_file(url, z)
        except Exception as e:
            print("maros: %s failed: %s" % (url, e))
            continue
        man.add("maros_meszaros/_download/" + os.path.basename(z), url, sha256_file(z))
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if not name or info.is_dir():
                    continue
                if name.upper().endswith(".QPS") or name.upper().startswith("00README"):
                    target = os.path.join(out, name.upper() if name.upper().endswith(".QPS") else name)
                    if not os.path.exists(target):
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    man.add("maros_meszaros/" + os.path.basename(target), url + " :: " + info.filename, sha256_file(target))
                    count += 1
    print("maros: %d files in %s" % (count, out))
    if verify and count:
        verify_maros(f, out)
    return count


def verify_maros(f, out):
    import json
    try:
        listing = json.loads(f.get(MAROS_MIRROR_API))
    except Exception as e:
        print("maros: cannot list the comparison repository: %s" % e)
        return
    same = diff = missing = 0
    for item in listing:
        n = item["name"]
        if not n.upper().endswith(".QPS"):
            continue
        local = os.path.join(out, n.upper())
        if not os.path.exists(local):
            missing += 1
            continue
        data = f.get(item["download_url"], timeout=600)
        with open(local, "rb") as fh:
            mine = fh.read()
        # Line endings and trailing blanks differ between the copies; the content must not.
        norm = lambda b: b"\n".join(l.rstrip() for l in b.replace(b"\r\n", b"\n").split(b"\n")).rstrip()
        if norm(data) == norm(mine):
            same += 1
        else:
            diff += 1
            print("maros: %s differs from the repository copy" % n)
    print("maros: comparison with YimingYAN/QP-Test-Problems (ignoring line endings and trailing blanks): %d identical, %d different, %d not in the official zips" % (same, diff, missing))


def fetch_miplib(f, data, man):
    out = os.path.join(data, "raw", "miplib")
    os.makedirs(out, exist_ok=True)
    z = os.path.join(out, "benchmark.zip")
    try:
        if not os.path.exists(z):
            f.to_file(MIPLIB, z, timeout=3600)
    except Exception as e:
        print("miplib: %s failed: %s (TLS verification is not bypassed; give --cafile)" % (MIPLIB, e))
        return 0
    man.add("miplib/benchmark.zip", MIPLIB, sha256_file(z))
    count = 0
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if not name or info.is_dir():
                continue
            target = os.path.join(out, name)
            if not os.path.exists(target):
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            man.add("miplib/" + name, MIPLIB + " :: " + info.filename, sha256_file(target))
            count += 1
    print("miplib: %d files in %s" % (count, out))
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="netlib,maros,miplib")
    ap.add_argument("--cafile", default=os.environ.get("SSL_CERT_FILE"))
    ap.add_argument("--verify-maros", action="store_true")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.data_dir, "raw"), exist_ok=True)
    man = Manifest(os.path.join(args.data_dir, "MANIFEST.txt"))
    f = Fetcher(args.cafile)
    which = set(args.only.split(","))
    if "netlib" in which:
        fetch_netlib(f, args.data_dir, man)
    if "maros" in which:
        fetch_maros(f, args.data_dir, man, args.verify_maros)
    if "miplib" in which:
        fetch_miplib(f, args.data_dir, man)
    man.save()
    print("manifest: %s (%d entries)" % (os.path.join(args.data_dir, "MANIFEST.txt"), len(man.rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
