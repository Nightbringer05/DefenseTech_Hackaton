#!/usr/bin/env python3
"""Local website for the drone sound toolkit + dataset builder.

    python server.py            # http://127.0.0.1:8770
    python server.py --port 9000 --host 0.0.0.0   # share on your LAN

Static pages work without this server too (open web/index.html); the builder then
falls back to printing the matching CLI command.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import shutil
import sys
import threading
import time
import traceback
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

BASE = Path(__file__).resolve().parent
WEB = BASE / "web"
sys.path.insert(0, str(BASE))

from dronesynth.dataset import CLASSES, DEFAULT_CONFIG, MAD_LABELS, build, estimate, merge_config, preview  # noqa: E402
from dronesynth.propagate import acoular_available  # noqa: E402
from dronesynth.drones import archetype_table  # noqa: E402
from dronesynth import spectro  # noqa: E402

OUT = BASE / "out"
PRESETS = BASE / "presets"


def _mad_counts() -> dict:
    import csv
    root = (BASE / DEFAULT_CONFIG["challenge_dir"] / "MAD_dataset").resolve()
    out: dict = {}
    for split in ("training", "test"):
        f = root / f"{split}.csv"
        if f.exists():
            with f.open() as fh:
                for r in csv.DictReader(fh):
                    d = out.setdefault(str(r["label"]), {"training": 0, "test": 0})
                    d[split] += 1
    return out


MAD_COUNTS = _mad_counts()
mimetypes.add_type("audio/flac", ".flac")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _run_job(jid: str, cfg: dict) -> None:
    def progress(done, total):
        with LOCK:
            JOBS[jid].update(done=done, total=total)
    try:
        m = build(cfg, OUT, BASE, progress)
        with LOCK:
            JOBS[jid].update(state="done", manifest=m, finished=time.time())
    except Exception as e:  # surface the error in the UI
        traceback.print_exc()
        with LOCK:
            JOBS[jid].update(state="error", error=f"{type(e).__name__}: {e}")


def load_presets() -> list[dict]:
    """synth/presets/*.json -> [{id, label, description, config}]; label/description keys are optional."""
    out = []
    for f in sorted(PRESETS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        label = d.pop("_label", None) or f.stem.replace("_", " ")
        desc = d.pop("_description", "")
        out.append({"id": f.stem, "label": label, "description": desc, "config": d})
    return out


def list_datasets() -> list[dict]:
    out = []
    if OUT.exists():
        for m in sorted(OUT.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(m.read_text())
            except Exception:
                continue
            samples = []
            lab = m.parent / "dataset" / "train" / "labels.jsonl"
            meta_f = m.parent / "dataset" / "train" / "metadata.jsonl"
            meta = {}
            if meta_f.exists():
                for line in meta_f.read_text().splitlines():
                    x = json.loads(line)
                    meta[x["id"]] = x
            seen = set()
            for line in lab.read_text().splitlines():
                r = json.loads(line)
                r.update({k: v for k, v in meta.get(r["id"], {}).items() if k != "id"})
                key = (r.get("source") or ("synthetic" if r.get("synthetic") else "challenge"), r["sound_class"])
                if key[0] != "challenge" and key not in seen:
                    seen.add(key)
                    samples.append({"id": r["id"], "class": r["sound_class"], "source": key[0],
                                    "audio": f"/files/{d['name']}/{r['audio']}",
                                    "png": f"/files/{d['name']}/{r['spectrogram']}"})
            d["audio_format"] = d.get("config", {}).get("audio_format", "wav")
            d["download_all"] = f"/download/{d['name']}.zip"
            d["download_audio"] = f"/download/{d['name']}.zip?parts=audio"
            d.pop("config", None)
            d["samples"] = samples
            d["path"] = str(m.parent.relative_to(BASE.parent)) if BASE.parent in m.parents else str(m.parent)
            out.append(d)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "DroneSynth/1.0"

    def log_message(self, fmt, *args):
        if "/api/jobs" not in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, root: Path, rel: str):
        p = (root / rel).resolve()
        if root.resolve() not in p.parents and p != root.resolve() or not p.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")

    def _zip(self, name: str, parts: str):
        """Zip a dataset. parts=audio -> audio files + labels; parts=all -> everything."""
        name = name.removesuffix(".zip")
        root = (OUT / name).resolve()
        if root.parent != OUT.resolve() or not (root / "manifest.json").exists():
            return self._send(404, b"dataset not found", "text/plain")
        parts = "audio" if parts == "audio" else "all"
        zpath = OUT / f".{name}.{parts}.zip"
        newest = max(f.stat().st_mtime for f in root.rglob("*") if f.is_file())
        if not zpath.exists() or zpath.stat().st_mtime < newest:
            tmp = zpath.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w") as z:
                for f in sorted(root.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(root)
                    if parts == "audio" and rel.parts[0] == "spectrograms":
                        continue
                    packed = f.suffix.lower() in (".flac", ".png")  # already compressed
                    z.write(f, f"{name}/{rel}", compress_type=zipfile.ZIP_STORED if packed else zipfile.ZIP_DEFLATED)
            tmp.replace(zpath)
        size = zpath.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(size))
        suffix = "" if parts == "all" else "_audio"
        self.send_header("Content-Disposition", f'attachment; filename="{name}{suffix}.zip"')
        self.end_headers()
        with zpath.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/status":
            return self._json({"server": True, "acoular": acoular_available(), "classes": CLASSES, "mad_labels": MAD_LABELS,
                               "mad_counts": MAD_COUNTS, "mad_found": bool(MAD_COUNTS),
                               "drone_types": archetype_table(),
                               "presets": load_presets(), "spectrogram_source": spectro.SOURCE,
                               "defaults": DEFAULT_CONFIG, "out_dir": str(OUT)})
        if path == "/api/jobs":
            with LOCK:
                return self._json(list(JOBS.values()))
        if path == "/api/datasets":
            return self._json(list_datasets())
        if path.startswith("/files/"):
            return self._file(OUT, path[len("/files/"):])
        if path.startswith("/download/"):
            return self._zip(path[len("/download/"):], parse_qs(urlparse(self.path).query).get("parts", ["all"])[0])
        rel = path.lstrip("/") or "index.html"
        return self._file(WEB, rel)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
        except Exception:
            return self._json({"error": "Request body is not valid JSON."}, 400)
        if path == "/api/preview":
            cfg = merge_config(body.get("config"))
            kind = body.get("kind", "drone_buzz")
            if kind not in CLASSES:
                return self._json({"error": f"Unknown class {kind!r}"}, 400)
            try:
                wav, png, params = preview(cfg, kind, BASE, source=body.get("source", "synthetic"))
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            fmt = "wav" if cfg.get("audio_format") == "wav" else "flac"
            return self._json({"audio": base64.b64encode(wav).decode(), "format": fmt,
                               "mime": f"audio/{fmt}", "png": base64.b64encode(png).decode(), "params": params})
        if path == "/api/estimate":
            try:
                return self._json(estimate(merge_config(body.get("config")), BASE))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/api/datasets":
            cfg = merge_config(body.get("config"))
            with LOCK:
                busy = [j for j in JOBS.values() if j["state"] == "running"]
                if busy:
                    return self._json({"error": "A dataset is already being generated. Wait for it to finish."}, 409)
                jid = uuid.uuid4().hex[:8]
                JOBS[jid] = {"id": jid, "name": cfg["name"], "state": "running", "done": 0, "total": 0,
                             "started": time.time()}
            threading.Thread(target=_run_job, args=(jid, cfg), daemon=True).start()
            return self._json({"id": jid}, 202)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Drone Sound Toolkit on http://{a.host}:{a.port}  (datasets -> {OUT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
