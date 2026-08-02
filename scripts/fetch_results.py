#!/usr/bin/env python3
"""Verify (or restore) the JAX-ESHN experiment-result data against the Zenodo mirror.

This repository commits its result data in-repo
(``papers/es-hyperneat-quadtree-problem/results/`` and ``benchmark_checkpoints/``), so
the paper's tables and figures can be reproduced offline with no download; see the
paper's ``scripts/runners/verify_*.py``. The same data is also deposited on Zenodo for archival and the
DOI the paper cites.

This script is therefore OPTIONAL. Use it to:

  * ``--check`` : verify a downloaded Zenodo archive against the pinned checksum, or
  * ``--restore`` : unpack the Zenodo archive into ``papers/*/`` (e.g. to restore an
    accidentally deleted ``results/`` directory).

Usage
-----
    python scripts/fetch_results.py --check                 # verify a local/downloaded archive
    python scripts/fetch_results.py --restore               # download + verify + unpack
    python scripts/fetch_results.py --archive PATH --restore # use a local archive
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# --- Data release coordinates -------------------------------------------------
ZENODO_DOI = "10.5281/zenodo.21761118"
ARCHIVE_NAME = "jax-es-hyperneat-data-v0.1.0.tar.gz"
ARCHIVE_URL = f"https://zenodo.org/records/21761118/files/{ARCHIVE_NAME}"
ARCHIVE_SHA256 = "9e9a809411de9dc0fe5067648780088dc1c5a058bcca36ed5f3ff3ac3ec75b10"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path) -> None:
    digest = _sha256(path)
    if digest != ARCHIVE_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {path.name}:\n  expected {ARCHIVE_SHA256}\n  got      {digest}"
        )
    print(f"  checksum OK ({digest[:16]}...)")


def _download(dest: Path) -> None:
    if "XXXXXXX" in ARCHIVE_URL:
        raise SystemExit(
            "The data-release URL is still a placeholder. The result data is already in-repo "
            "under papers/*/results and papers/*/benchmark_checkpoints; no download is needed "
            "to reproduce the paper. Set ZENODO_DOI/ARCHIVE_URL once the archive is uploaded, or "
            f"point at a local copy: python scripts/fetch_results.py --archive <{ARCHIVE_NAME}> --check"
        )
    print(f"Downloading {ARCHIVE_URL}")
    urllib.request.urlretrieve(ARCHIVE_URL, dest)  # noqa: S310 (trusted Zenodo URL)


def _unpack(archive: Path) -> None:
    print(f"Unpacking into {REPO_ROOT}")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("papers/")]
        for m in members:  # guard against path traversal
            target = (REPO_ROOT / m.name).resolve()
            if not str(target).startswith(str(REPO_ROOT)):
                raise SystemExit(f"Refusing to extract outside the repo: {m.name}")
        try:
            tar.extractall(REPO_ROOT, members=members, filter="data")
        except TypeError:  # filter= added in Python 3.12
            tar.extractall(REPO_ROOT, members=members)
    print(f"  unpacked {len(members)} entries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify a local/downloaded archive only")
    ap.add_argument("--restore", action="store_true", help="download/unpack the archive into papers/*/")
    ap.add_argument("--archive", type=Path, default=None, help="path to a local archive")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded archive")
    args = ap.parse_args()

    print(f"JAX-ESHN data mirror ({ARCHIVE_NAME}, DOI {ZENODO_DOI})")
    print("Note: result data is committed in-repo; this Zenodo mirror is optional.")

    if args.archive:
        print(f"Using local archive {args.archive}")
        _verify(args.archive)
        if args.restore:
            _unpack(args.archive)
        return 0

    tmp = Path(tempfile.gettempdir()) / ARCHIVE_NAME
    if args.check and not args.restore:
        if not tmp.exists():
            raise SystemExit(f"No local archive at {tmp}; pass --archive PATH or --restore.")
        _verify(tmp)
        return 0

    _download(tmp)
    print("Verifying checksum...")
    _verify(tmp)
    _unpack(tmp)
    if not args.keep:
        tmp.unlink(missing_ok=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
