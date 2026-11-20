#!/usr/bin/env python3
"""Zero-dependency local web viewer for Challenge 5: Acoustic Cue-to-Optical Slew C-UAS."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SOUND_CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")

_EXTRA_MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
}

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Challenge 5 Viewer — Acoustic Cue-to-Optical Slew C-UAS</title>
<style>
  :root {
    --bg: #0b1117;
    --panel: #131c26;
    --panel-alt: #182330;
    --border: #273647;
    --text: #e6edf3;
    --muted: #8b9bb0;
    --accent: #38bdf8;
    --wake: #f59e0b;
    --good: #10b981;
    --danger: #f43f5e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .container { padding: 16px 20px; max-width: 1560px; margin: 0 auto; }
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
  }
  .stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .stat-val { font-size: 19px; font-weight: 700; margin-top: 3px; }
  .banner {
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: #fde68a;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 13px;
    display: none;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 14px;
  }
  .toolbar label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  select, input[type="text"] {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 13px;
  }
  button.btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 6px 11px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }
  button.btn.primary { background: var(--accent); color: #062129; border-color: var(--accent); }
  .split-layout {
    display: grid;
    grid-template-columns: minmax(290px, 0.9fr) minmax(420px, 1.4fr);
    gap: 14px;
    align-items: start;
  }
  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 4px;
  }
  .thumb-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px;
    cursor: pointer;
  }
  .thumb-card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .thumb-card img {
    width: 100%;
    height: 92px;
    object-fit: cover;
    border-radius: 4px;
    background: #05080c;
    display: block;
  }
  .thumb-meta {
    margin-top: 5px;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    gap: 4px;
    color: var(--muted);
  }
  .badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--panel-alt);
    color: var(--text);
  }
  .badge.wake { border-color: var(--wake); color: var(--wake); }
  .badge.quiet { border-color: var(--accent); color: var(--accent); }
  .badge.good { border-color: var(--good); color: var(--good); }
  .badge.bad { border-color: var(--danger); color: var(--danger); }
  .inspector {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .triplet-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }
  .media-box {
    background: #05080c;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
  }
  .media-box h3 { margin: 0 0 6px 0; font-size: 12px; color: var(--muted); text-transform: uppercase; }
  video { width: 100%; max-height: 36vh; border-radius: 4px; display: block; background: #000; }
  canvas { width: 100%; max-height: 36vh; border-radius: 4px; display: block; background: #000; }
  audio { width: 100%; margin-top: 8px; }
  .pager { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Challenge 5: Acoustic Cue-to-Optical Slew C-UAS Explorer</h1>
  <div class="sub">Synchronized 5s 16 kHz Field WAV + 30 fps CFR MP4 Video + 224x224 Log-Mel Spectrogram &middot; 4 Sound Classes</div>
</header>

<div class="container">
  <div class="banner" id="gtHiddenBanner">Held-out <code>test</code> split: acoustic wake &amp; sound-class ground truth is stripped in public archives so teams score locally on <code>valid</code>.</div>
  <div class="stats-row" id="statsRow"></div>

  <div class="toolbar">
    <label>Split <select id="splitSelect"></select></label>
    <label>Wake Gate
      <select id="wakeFilter">
        <option value="all">All clips</option>
        <option value="true">Wake = True (drone_audio)</option>
        <option value="false">Wake = False</option>
      </select>
    </label>
    <label>Sound Class
      <select id="classFilter">
        <option value="all">All 4 classes</option>
        <option value="drone_buzz">drone_buzz</option>
        <option value="vehicle_mechanical">vehicle_mechanical</option>
        <option value="explosion">explosion</option>
        <option value="ambient_outdoor">ambient_outdoor</option>
      </select>
    </label>
    <label>Pred Analysis
      <select id="predFilter">
        <option value="all">All samples</option>
        <option value="fp">Wake False Positives</option>
        <option value="fn">Wake False Negatives</option>
        <option value="cls_err">Sound Class Mismatch</option>
      </select>
    </label>
    <label>Search <input id="searchBox" type="text" placeholder="c5_0001" /></label>
  </div>

  <div class="split-layout">
    <div>
      <div class="thumb-grid" id="thumbGrid"></div>
      <div class="pager">
        <button class="btn" id="prevPage" type="button">Prev</button>
        <span id="pageInfo">Page 1</span>
        <button class="btn" id="nextPage" type="button">Next</button>
      </div>
    </div>

    <div class="inspector">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <span id="inspTitle" style="font-weight:700;font-size:15px;"></span>
          <span id="inspBadges" style="margin-left:8px;"></span>
        </div>
        <div id="syncTransport" style="display:flex;gap:8px;">
          <button class="btn primary" id="playSyncBtn" type="button">Play Synced (Video + 16 kHz Audio)</button>
          <button class="btn" id="resetSyncBtn" type="button">Restart 0.0s</button>
        </div>
      </div>

      <div class="triplet-grid">
        <div class="media-box" id="videoBox">
          <h3>Paired Optical FOV (30 fps CFR, Muted Sync)</h3>
          <video id="tripVideo" controls muted playsinline preload="metadata"></video>
        </div>
        <div class="media-box">
          <h3>Log-Mel Spectrogram (224x224 &middot; Live Playhead)</h3>
          <canvas id="specCanvas" width="448" height="448"></canvas>
        </div>
      </div>

      <div class="media-box" id="audioBox" style="margin-top:12px;">
        <h3>16 kHz Mono PCM Field Microphone Audio</h3>
        <audio id="tripAudio" controls preload="metadata"></audio>
      </div>

      <div id="predCompare" style="margin-top:10px;font-size:13px;color:var(--muted);"></div>
    </div>
  </div>
</div>

<script>
(() => {
  const PAGE_SIZE = 32;
  let state = null;
  let filteredItems = [];
  let page = 0;
  let selectedId = null;
  let activeSpecImg = null;

  function el(id) { return document.getElementById(id); }

  function mediaUrl(rel) {
    return "/media/" + String(rel).split("/").map(encodeURIComponent).join("/");
  }

  function populateSplits() {
    const sel = el("splitSelect");
    sel.innerHTML = "";
    const order = ["valid", "train", "test", "calibration_spectrograms"];
    const keys = Object.keys(state.splits).sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    for (const k of keys) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${k} (${state.splits[k].counts.sample_count})`;
      sel.appendChild(opt);
    }
  }

  function renderStats(splitObj) {
    const c = splitObj.counts;
    const row = el("statsRow");
    row.innerHTML = "";
    const cards = [
      ["Samples", c.sample_count],
      ["Drone Wake (True / False)", splitObj.gt_available ? `${c.wake_true} / ${c.wake_false}` : "Hidden"],
      ["drone / vehicle", splitObj.gt_available ? `${c.class_counts.drone_buzz || 0} / ${c.class_counts.vehicle_mechanical || 0}` : "—"],
      ["explosion / ambient", splitObj.gt_available ? `${c.class_counts.explosion || 0} / ${c.class_counts.ambient_outdoor || 0}` : "—"],
      ["Wake F1 (--preds)", c.wake_f1 != null ? c.wake_f1.toFixed(4) : "N/A"],
      ["Energy Saved / Duty", c.energy_savings_pct != null ? `${c.energy_savings_pct.toFixed(1)}% (FP:${c.fp_count}, FN:${c.fn_count})` : "N/A"],
    ];
    for (const [lbl, val] of cards) {
      const d = document.createElement("div");
      d.className = "stat-card";
      const l = document.createElement("div");
      l.className = "stat-label";
      l.textContent = lbl;
      const v = document.createElement("div");
      v.className = "stat-val";
      v.textContent = String(val);
      d.appendChild(l);
      d.appendChild(v);
      row.appendChild(d);
    }
    el("gtHiddenBanner").style.display = (!splitObj.gt_available && el("splitSelect").value === "test") ? "block" : "none";
  }

  function applyFilters() {
    const splitObj = state.splits[el("splitSelect").value];
    if (!splitObj) return;
    renderStats(splitObj);

    const wakeVal = el("wakeFilter").value;
    const clsVal = el("classFilter").value;
    const predVal = el("predFilter").value;
    const q = el("searchBox").value.trim().toLowerCase();

    filteredItems = splitObj.items.filter((it) => {
      if (q && !it.id.toLowerCase().includes(q)) return false;
      if (wakeVal === "true" && it.has_drone_audio !== true) return false;
      if (wakeVal === "false" && it.has_drone_audio !== false) return false;
      if (clsVal !== "all" && it.sound_class !== clsVal) return false;
      if (predVal === "fp" && !(it.pred_wake === true && it.has_drone_audio === false)) return false;
      if (predVal === "fn" && !(it.pred_wake === false && it.has_drone_audio === true)) return false;
      if (predVal === "cls_err" && !(it.pred_sound_class && it.sound_class && it.pred_sound_class !== it.sound_class)) return false;
      return true;
    });

    page = 0;
    renderPage();
  }

  function renderPage() {
    const grid = el("thumbGrid");
    grid.innerHTML = "";
    const totalPages = Math.max(1, Math.ceil(filteredItems.length / PAGE_SIZE));
    page = Math.min(Math.max(0, page), totalPages - 1);
    const slice = filteredItems.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
    el("pageInfo").textContent = `Page ${page + 1} / ${totalPages} (${filteredItems.length} clips)`;

    if (slice.length > 0 && !slice.some((x) => x.id === selectedId)) {
      selectedId = slice[0].id;
    }

    const splitObj = state.splits[el("splitSelect").value];
    const gtAvail = Boolean(splitObj && splitObj.gt_available);

    for (const it of slice) {
      const card = document.createElement("div");
      card.className = "thumb-card" + (it.id === selectedId ? " selected" : "");
      card.onclick = () => { selectedId = it.id; renderPage(); renderInspector(it); };

      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = mediaUrl(it.spectrogram_src);
      img.alt = it.id;
      card.appendChild(img);

      const meta = document.createElement("div");
      meta.className = "thumb-meta";
      const s1 = document.createElement("span");
      s1.textContent = it.id;
      const b = document.createElement("span");
      if (!gtAvail || it.has_drone_audio === null) {
        b.className = "badge";
        b.textContent = it.pred_sound_class ? `pred:${it.pred_sound_class}` : "hidden";
      } else {
        b.className = "badge " + (it.has_drone_audio ? "wake" : "quiet");
        b.textContent = it.sound_class || "?";
      }
      meta.appendChild(s1);
      meta.appendChild(b);
      card.appendChild(meta);
      grid.appendChild(card);
    }

    const active = slice.find((x) => x.id === selectedId) || slice[0];
    if (active) renderInspector(active);
  }

  function drawSpectrogramPlayhead() {
    const canvas = el("specCanvas");
    const ctx = canvas.getContext("2d");
    const audio = el("tripAudio");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (activeSpecImg && activeSpecImg.complete && activeSpecImg.naturalWidth > 0) {
      ctx.drawImage(activeSpecImg, 0, 0, canvas.width, canvas.height);
    }
    const dur = (audio && Number.isFinite(audio.duration) && audio.duration > 0) ? audio.duration : 5.0;
    const t = audio ? audio.currentTime : 0;
    const frac = Math.max(0, Math.min(1, t / dur));
    const px = Math.round(frac * canvas.width);

    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();

    ctx.fillStyle = "rgba(11,17,23,0.82)";
    ctx.fillRect(6, 6, 118, 22);
    ctx.fillStyle = "#6ee7b7";
    ctx.font = "bold 13px ui-monospace, monospace";
    ctx.fillText(`t = ${t.toFixed(2)}s / ${dur.toFixed(1)}s`, 12, 21);
  }

  function renderInspector(it) {
    const splitObj = state.splits[el("splitSelect").value];
    const gtAvail = Boolean(splitObj && splitObj.gt_available);
    el("inspTitle").textContent = it.id;
    const bWrap = el("inspBadges");
    bWrap.innerHTML = "";
    if (gtAvail && it.has_drone_audio !== null) {
      const bw = document.createElement("span");
      bw.className = "badge " + (it.has_drone_audio ? "wake" : "quiet");
      bw.textContent = it.has_drone_audio ? "GT WAKE: TRUE" : "GT WAKE: FALSE";
      bWrap.appendChild(bw);

      const bc = document.createElement("span");
      bc.className = "badge";
      bc.style.marginLeft = "6px";
      bc.textContent = String(it.sound_class || "");
      bWrap.appendChild(bc);
    } else {
      const bh = document.createElement("span");
      bh.className = "badge";
      bh.textContent = it.split === "calibration_spectrograms" ? "Train-only Calibration" : "Held-out Test";
      bWrap.appendChild(bh);
    }

    const vid = el("tripVideo");
    const aud = el("tripAudio");
    const hasAV = Boolean(it.video_src && it.audio_src);
    el("videoBox").style.display = hasAV ? "block" : "none";
    el("audioBox").style.display = hasAV ? "block" : "none";
    el("syncTransport").style.display = hasAV ? "flex" : "none";

    if (hasAV) {
      vid.pause();
      aud.pause();
      vid.src = mediaUrl(it.video_src);
      aud.src = mediaUrl(it.audio_src);
      vid.load();
      aud.load();
    } else {
      vid.removeAttribute("src");
      aud.removeAttribute("src");
    }

    const specImg = new Image();
    specImg.onload = () => { activeSpecImg = specImg; drawSpectrogramPlayhead(); };
    specImg.src = mediaUrl(it.spectrogram_src);

    if (it.pred_wake !== null || it.pred_sound_class !== null) {
      const wakeMatch = it.has_drone_audio === null ? "N/A" : (it.pred_wake === it.has_drone_audio ? "MATCH" : "MISMATCH");
      el("predCompare").textContent =
        `Model Prediction -> wake_triggered=${it.pred_wake}, sound_class=${it.pred_sound_class} (Wake vs GT: ${wakeMatch})`;
    } else {
      el("predCompare").textContent = "Pass --preds <ac_preds.jsonl> to overlay acoustic wake predictions and energy savings.";
    }
  }

  async function init() {
    const resp = await fetch("/api/state");
    state = await resp.json();
    populateSplits();
    applyFilters();

    for (const id of ["splitSelect", "wakeFilter", "classFilter", "predFilter"]) {
      el(id).onchange = applyFilters;
    }
    el("searchBox").oninput = applyFilters;
    el("prevPage").onclick = () => { if (page > 0) { page--; renderPage(); } };
    el("nextPage").onclick = () => { page++; renderPage(); };

    const vid = el("tripVideo");
    const aud = el("tripAudio");
    el("playSyncBtn").onclick = () => {
      if (aud.paused) {
        vid.currentTime = aud.currentTime;
        aud.play();
        vid.play();
      } else {
        aud.pause();
        vid.pause();
      }
    };
    el("resetSyncBtn").onclick = () => {
      aud.currentTime = 0;
      vid.currentTime = 0;
      drawSpectrogramPlayhead();
    };
    aud.addEventListener("timeupdate", drawSpectrogramPlayhead);
    aud.addEventListener("seeked", () => {
      if (Math.abs(vid.currentTime - aud.currentTime) > 0.15) {
        vid.currentTime = aud.currentTime;
      }
      drawSpectrogramPlayhead();
    });
    function tick() {
      if (!aud.paused && !aud.ended) drawSpectrogramPlayhead();
      window.requestAnimationFrame(tick);
    }
    window.requestAnimationFrame(tick);
  }

  init();
})();
</script>
</body>
</html>
"""


def _load_acoustic_preds(preds_path: Path | None) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    if preds_path is None or not preds_path.is_file():
        return preds
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row["id"])
        preds[item_id] = {
            "wake_triggered": row["wake_triggered"]
            if isinstance(row.get("wake_triggered"), bool)
            else None,
            "sound_class": row["sound_class"]
            if isinstance(row.get("sound_class"), str)
            else None,
        }
    return preds


def build_state(
    challenge_dir: Path,
    *,
    preds_path: Path | None = None,
) -> dict[str, Any]:
    """Index Challenge 5 synchronized audio/video/spectrogram splits and calibration PNGs."""
    dataset_dir = challenge_dir / "dataset"
    preds_by_id = _load_acoustic_preds(preds_path)
    splits: dict[str, Any] = {}

    for split in ("train", "valid", "test"):
        jsonl_path = dataset_dir / split / "labels.jsonl"
        if not jsonl_path.is_file():
            continue
        raw_rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gt_available = any(
            "has_drone_audio" in row and "sound_class" in row for row in raw_rows
        )
        items: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {name: 0 for name in SOUND_CLASSES}
        wake_true = 0
        wake_false = 0

        for row in raw_rows:
            item_id = str(row["id"])
            wake_gt = bool(row["has_drone_audio"]) if "has_drone_audio" in row else None
            cls_gt = str(row["sound_class"]) if "sound_class" in row else None
            if wake_gt is True:
                wake_true += 1
            elif wake_gt is False:
                wake_false += 1
            if cls_gt is not None:
                class_counts[cls_gt] = class_counts.get(cls_gt, 0) + 1
            pred = preds_by_id.get(item_id, {})
            items.append(
                {
                    "id": item_id,
                    "split": split,
                    "audio_src": row["audio"],
                    "video_src": row["video"],
                    "spectrogram_src": row["spectrogram"],
                    "has_drone_audio": wake_gt,
                    "sound_class": cls_gt,
                    "pred_wake": pred.get("wake_triggered"),
                    "pred_sound_class": pred.get("sound_class"),
                }
            )

        wake_f1: float | None = None
        sound_class_accuracy: float | None = None
        optical_duty_cycle: float | None = None
        energy_savings_pct: float | None = None
        fp_count: int | None = None
        fn_count: int | None = None

        matched = [
            it
            for it in items
            if it["pred_wake"] is not None and it["has_drone_audio"] is not None
        ]
        if gt_available and matched and len(matched) == len(items):
            tp = sum(
                1
                for it in matched
                if it["pred_wake"] is True and it["has_drone_audio"] is True
            )
            fp = sum(
                1
                for it in matched
                if it["pred_wake"] is True and it["has_drone_audio"] is False
            )
            fn = sum(
                1
                for it in matched
                if it["pred_wake"] is False and it["has_drone_audio"] is True
            )
            wake_f1 = round((2.0 * tp) / max(1, 2 * tp + fp + fn), 4)
            duty = sum(1 for it in matched if it["pred_wake"] is True) / len(matched)
            cls_acc = sum(
                1 for it in matched if it["pred_sound_class"] == it["sound_class"]
            ) / len(matched)
            optical_duty_cycle = round(duty, 4)
            energy_savings_pct = round((1.0 - duty) * 100.0, 2)
            sound_class_accuracy = round(cls_acc, 4)
            fp_count = fp
            fn_count = fn

        splits[split] = {
            "gt_available": gt_available,
            "counts": {
                "sample_count": len(items),
                "wake_true": wake_true,
                "wake_false": wake_false,
                "class_counts": class_counts,
                "wake_f1": wake_f1,
                "sound_class_accuracy": sound_class_accuracy,
                "optical_duty_cycle": optical_duty_cycle,
                "energy_savings_pct": energy_savings_pct,
                "fp_count": fp_count,
                "fn_count": fn_count,
            },
            "items": items,
        }

    cal_dir = challenge_dir / "calibration_spectrograms"
    if cal_dir.is_dir():
        cal_files = sorted(
            p
            for p in cal_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        cal_items = [
            {
                "id": p.stem,
                "split": "calibration_spectrograms",
                "audio_src": None,
                "video_src": None,
                "spectrogram_src": f"calibration_spectrograms/{p.name}",
                "has_drone_audio": None,
                "sound_class": None,
                "pred_wake": None,
                "pred_sound_class": None,
            }
            for p in cal_files
        ]
        splits["calibration_spectrograms"] = {
            "gt_available": False,
            "counts": {
                "sample_count": len(cal_items),
                "wake_true": 0,
                "wake_false": 0,
                "class_counts": {},
                "wake_f1": None,
                "sound_class_accuracy": None,
                "optical_duty_cycle": None,
                "energy_savings_pct": None,
                "fp_count": None,
                "fn_count": None,
            },
            "items": cal_items,
        }

    return {
        "challenge": "challenge-5-acoustic-visual-cuas",
        "title": "Challenge 5: Acoustic Cue-to-Optical Slew C-UAS",
        "sound_classes": list(SOUND_CLASSES),
        "splits": splits,
    }


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _resolve_safe_media_path(root_dir: Path, raw_rel: str) -> Path | None:
    rel = urllib.parse.unquote(raw_rel).lstrip("/")
    if not rel or "\x00" in rel:
        return None
    root_resolved = root_dir.resolve()
    candidate = (root_resolved / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _serve_file_with_range(
    handler: BaseHTTPRequestHandler,
    path: Path,
    *,
    send_body: bool = True,
) -> None:
    size = path.stat().st_size
    mime = _guess_mime(path)
    range_hdr = handler.headers.get("Range")
    if range_hdr and range_hdr.startswith("bytes="):
        spec = range_hdr[len("bytes=") :].split(",", 1)[0].strip()
        start_str, _, end_str = spec.partition("-")
        try:
            if start_str == "":
                suffix_len = int(end_str)
                start = max(0, size - suffix_len)
                end = size - 1
            else:
                start = int(start_str)
                end = int(end_str) if end_str else size - 1
        except ValueError:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        if size == 0 or start < 0 or end < start or start >= size:
            handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
        end = min(end, size - 1)
        length = end - start + 1
        handler.send_response(HTTPStatus.PARTIAL_CONTENT)
        handler.send_header("Content-Type", mime)
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.send_header("Content-Length", str(length))
        handler.end_headers()
        if not send_body:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
        return

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(size))
    handler.end_headers()
    if not send_body:
        return
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            handler.wfile.write(chunk)


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(
    challenge_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8768,
    preds_path: Path | None = None,
) -> ThreadingHTTPServer:
    """Create the Challenge 5 HTTP viewer server bound to ``host:port``."""
    root_dir = challenge_dir.resolve()
    state_bytes = json.dumps(
        build_state(root_dir, preds_path=preds_path), allow_nan=False
    ).encode("utf-8")
    html_bytes = _HTML_PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_request(self, *, send_body: bool) -> None:
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            try:
                if route in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(html_bytes)
                    return

                if route == "/api/state":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(state_bytes)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(state_bytes)
                    return

                if route.startswith("/media/"):
                    target = _resolve_safe_media_path(root_dir, route[len("/media/") :])
                    if target is None:
                        self.send_error(HTTPStatus.FORBIDDEN)
                        return
                    _serve_file_with_range(self, target, send_body=send_body)
                    return

                self.send_error(HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self) -> None:
            self._handle_request(send_body=True)

        def do_HEAD(self) -> None:
            self._handle_request(send_body=False)

    return _QuietThreadingHTTPServer((host, port), Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Challenge 5 directory containing audio/, video/, spectrograms/, dataset/",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--preds",
        type=Path,
        default=None,
        help="Optional ac_preds.jsonl to overlay wake predictions & energy savings",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = create_server(
        args.challenge_dir,
        host=args.host,
        port=args.port,
        preds_path=args.preds,
    )
    actual_url = f"http://{args.host}:{server.server_port}/"
    print(f"Challenge 5 viewer listening on {actual_url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(actual_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
