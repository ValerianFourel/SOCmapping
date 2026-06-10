#!/usr/bin/env python3
"""verify_hf_match.py — verify a local Data/ tree against its HuggingFace origin.

Compares a local copy (default: ./Data) against a canonical HF dataset
(default: ValerianFourel/sgt-bavaria-soc-2002-2023) and reports whether they
match:
  * file set        — files on HF but missing locally, and local-only extras
  * sizes           — per-file byte size (fast; catches truncated/partial pulls)
  * content hashes  — with --hash: LFS files by sha256, plain git files by
                      git-blob sha1 (authoritative; reads every byte)

It also prints the recorded *origin commit* of the local copy if HF download
metadata (.cache/huggingface/download/**/*.metadata) is present, so you can
tell which HF revision the data was pulled from.

Run with your env active (huggingface_hub importable):
    python verify_hf_match.py --data /hkfs/.../SGT/SOCmapping/Data
    python verify_hf_match.py --data ... --hash --workers 16     # full integrity
    python verify_hf_match.py --data ... --repo ValerianFourel/SOCmappingRastersAndSoilSamples
                                                                 # compare vs the OLD repo

Exit code 0 == match, 1 == mismatch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_DEFAULT = "ValerianFourel/sgt-bavaria-soc-2002-2023"
IGNORE_HF = {".gitattributes", ".gitignore"}            # repo meta, never in the local tree
LOCAL_ONLY = {"Preprocessing", "RasterBandsData",       # top-level trees the publisher drops
              "pipeline_state.json", ".cache"}


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def hash_file(p, algo):
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1()
    if algo == "sha1":                                  # git blob sha1 = sha1("blob <len>\0" + bytes)
        h.update(f"blob {os.path.getsize(p)}\0".encode())
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_origin(data):
    """Best-effort: commit hashes recorded by snapshot_download in local_dir."""
    base = Path(data) / ".cache" / "huggingface" / "download"
    commits, n = set(), 0
    if base.is_dir():
        for m in base.rglob("*.metadata"):
            n += 1
            try:
                commits.add(m.read_text().splitlines()[0].strip())
            except Exception:
                pass
    return commits, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="Data", help="local data dir (default: Data)")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--hash", action="store_true", help="verify content hashes (reads every byte)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None, help="also write a JSON report to this path")
    a = ap.parse_args()

    data = Path(a.data).resolve()
    if not data.is_dir():
        sys.exit(f"[error] no such dir: {data}")

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"[verify] querying HF: {a.repo} (rev={a.revision or 'main'}) …", flush=True)
    info = api.repo_info(a.repo, repo_type="dataset", revision=a.revision, files_metadata=True)
    head = info.sha

    # HF index: path -> (size, algo, expected_hash)
    hf = {}
    for s in info.siblings or []:
        rf = s.rfilename
        if rf in IGNORE_HF:
            continue
        lfs = getattr(s, "lfs", None)
        if lfs:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            hf[rf] = (s.size, "sha256", sha)
        else:
            hf[rf] = (s.size, "sha1", getattr(s, "blob_id", None))

    # Local index (skip the local-only top-level trees and any .cache dir)
    local = {}
    for root, dirs, files in os.walk(data):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for fn in files:
            fp = Path(root) / fn
            rel = os.path.relpath(fp, data)
            if rel.split(os.sep)[0] in LOCAL_ONLY:
                continue
            local[rel] = fp

    hf_set, loc_set = set(hf), set(local)
    missing = sorted(hf_set - loc_set)      # on HF, absent locally
    extra = sorted(loc_set - hf_set)        # local-only
    common = sorted(hf_set & loc_set)

    size_bad, ok_common = [], []
    for rel in common:
        exp = hf[rel][0]
        act = local[rel].stat().st_size
        if exp is not None and act != exp:
            size_bad.append((rel, exp, act))
        else:
            ok_common.append(rel)

    hash_bad, checked, bytes_done = [], 0, 0
    if a.hash and ok_common:
        total = len(ok_common)
        tbytes = sum(hf[r][0] or 0 for r in ok_common)
        print(f"[verify] hashing {total} files ({human(tbytes)}) with {a.workers} workers …", flush=True)

        def chk(rel):
            _, algo, exp = hf[rel]
            if not exp:
                return rel, None
            return rel, (hash_file(local[rel], algo) == exp)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(chk, r): r for r in ok_common}
            for i, fut in enumerate(as_completed(futs), 1):
                rel, good = fut.result()
                checked += 1
                bytes_done += hf[rel][0] or 0
                if good is False:
                    hash_bad.append(rel)
                if i % 500 == 0 or i == total:
                    print(f"    {i}/{total}  ({human(bytes_done)})  mismatches={len(hash_bad)}", flush=True)

    commits, nmeta = read_origin(data)

    print("\n================ HF ORIGIN ================")
    print(f"repo        : {a.repo}")
    print(f"HF HEAD sha : {head}")
    if nmeta:
        same = commits == {head}
        print(f"local pulled: {', '.join(sorted(commits))}  ({nmeta} metadata files)")
        print("            : " + ("== HF HEAD  (same revision)" if same
                                   else "!= HF HEAD  (HF moved since this copy was pulled)"))
    else:
        print("local pulled: no .cache/huggingface metadata (origin commit unknown; judged by content below)")

    print("\n================ COMPARISON ===============")
    print(f"HF files       : {len(hf_set):6d}   ({human(sum(v[0] or 0 for v in hf.values()))})")
    print(f"local files    : {len(loc_set):6d}")
    print(f"common         : {len(common):6d}")
    print(f"missing locally: {len(missing):6d}   (on HF, absent here)")
    print(f"extra locally  : {len(extra):6d}   (here, not on HF)")
    print(f"size mismatch  : {len(size_bad):6d}")
    if a.hash:
        print(f"hash mismatch  : {len(hash_bad):6d}   (checked {checked})")
    else:
        print("hash check     : SKIPPED (size-only; add --hash for full integrity)")

    def show(title, items, key=str, n=12):
        if items:
            print(f"\n-- {title} ({len(items)}) --")
            for x in items[:n]:
                print("   ", key(x))
            if len(items) > n:
                print(f"    … +{len(items) - n} more")

    show("MISSING locally", missing)
    show("EXTRA locally", extra)
    show("SIZE mismatch  (path | hf | local)", size_bad, key=lambda t: f"{t[0]} | {t[1]} | {t[2]}")
    if a.hash:
        show("HASH mismatch", hash_bad)

    match = not (missing or extra or size_bad or hash_bad)
    level = "content (sha)" if a.hash else "size-only"
    print("\n================ VERDICT ==================")
    print(("MATCH  — local == HF  [" + level + "]") if match
          else "MISMATCH — local != HF  (see lists above)")
    print("==========================================")

    if a.json:
        json.dump(dict(repo=a.repo, head=head, origin=sorted(commits),
                       hf_files=len(hf_set), local_files=len(loc_set),
                       missing=missing, extra=extra,
                       size_mismatch=size_bad, hash_mismatch=hash_bad,
                       hashed=a.hash, match=match),
                  open(a.json, "w"), indent=2)
        print(f"[verify] wrote {a.json}")

    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
