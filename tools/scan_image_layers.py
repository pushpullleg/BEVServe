"""Scan every layer of a saved Docker image for restricted artifacts.

    docker save IMAGE | python3 tools/scan_image_layers.py --checkpoint-size N --checkpoint-sha256 H

Reads the `docker save` tar stream from stdin (Docker or OCI layout), opens every layer tar inside
it (files deleted in later layers are still present in earlier ones) and flags:
- a file whose size equals the checkpoint's and whose sha256 matches it (the exact weights);
- checkpoint-like files: *.pth over 1 MiB (small *.pth files are Python path files), *.ckpt,
  *.pt / *.safetensors over 1 MiB;
- nuScenes-like content: */samples/CAM_*, */sweeps/*, v1.0-trainval / v1.0-mini trees, *.pcd.bin,
  *infos*.pkl;
- private paths baked into file names (home/<user>/...).

Exit 0 only if the stream was read to the end, at least one layer was scanned and nothing was
flagged; any read error or finding exits non-zero. Prints one JSON summary line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from typing import IO, Any

MiB = 1024 * 1024
PATTERNS = {
    "nuscenes-camera-sample": re.compile(r"(^|/)samples/CAM_"),
    "nuscenes-sweep": re.compile(r"(^|/)sweeps/(CAM|LIDAR|RADAR)_"),
    "nuscenes-tables": re.compile(r"(^|/)v1\.0-(trainval|mini|test)(/|$)"),
    "lidar-point-cloud": re.compile(r"\.pcd\.bin$"),
    "dataset-info-pickle": re.compile(r"infos?[^/]*\.pkl$"),
    "home-directory-path": re.compile(r"(^|/)home/[^/]+/"),
    "checkpoint-ckpt": re.compile(r"\.ckpt$"),
}
LARGE_WEIGHT = re.compile(r"\.(pth|pt|safetensors)$")


def scan_layer(layer: IO[bytes], ckpt_size: int, ckpt_sha: str, label: str) -> dict[str, Any]:
    files, findings = 0, []
    with tarfile.open(fileobj=layer, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            files += 1
            name = member.name
            for kind, pattern in PATTERNS.items():
                if pattern.search(name):
                    findings.append({"layer": label, "path": name, "kind": kind})
            if LARGE_WEIGHT.search(name) and member.size > MiB:
                findings.append({"layer": label, "path": name, "kind": "large-weight-file"})
            if ckpt_size and member.size == ckpt_size:
                stream = tar.extractfile(member)
                digest = hashlib.sha256()
                while stream is not None and (chunk := stream.read(MiB)):
                    digest.update(chunk)
                if digest.hexdigest() == ckpt_sha:
                    findings.append({"layer": label, "path": name, "kind": "checkpoint-bytes"})
    return {"files": files, "findings": findings}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-size", type=int, default=0)
    p.add_argument("--checkpoint-sha256", default="")
    args = p.parse_args(argv)
    layers, files, findings = 0, 0, []
    try:
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as outer:
            for member in outer:
                if not member.isfile() or member.size < 1024:
                    continue
                name = member.name
                if not (name.endswith("layer.tar") or name.startswith("blobs/")):
                    continue
                fileobj = outer.extractfile(member)
                if fileobj is None:
                    continue
                try:
                    result = scan_layer(fileobj, args.checkpoint_size, args.checkpoint_sha256, name)
                except tarfile.ReadError:
                    continue  # a config/manifest blob, not a layer
                layers += 1
                files += result["files"]
                findings += result["findings"]
    except (tarfile.TarError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    ok = layers > 0 and not findings
    print(json.dumps({"ok": ok, "layers": layers, "files": files, "findings": findings[:50]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
