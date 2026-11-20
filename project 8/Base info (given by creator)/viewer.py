#!/usr/bin/env python3
"""Zero-dependency local web viewer for Challenge 4: Edge Bandwidth Triage Distillation."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCENE_KEYS = (
    "armored_vehicle",
    "vehicle",
    "person",
    "drone",
    "explosion_smoke",
    "trench_building",
)
MODALITIES = ("rgb", "night", "thermal")
PERSPECTIVES = ("from_drone", "ground", "of_drone", "other")

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
<title>Challenge 4 Viewer — Edge Bandwidth Triage Distillation</title>
<style>
  :root {
    --bg: #0b1117;
    --panel: #131c26;
    --panel-alt: #182330;
    --border: #273647;
    --text: #e6edf3;
    --muted: #8b9bb0;
    --accent: #34d399;
    --warn: #f59e0b;
    --danger: #f43f5e;
    --info: #38bdf8;
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
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
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
    margin-bottom: 10px;
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
  .tag-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 14px;
  }
  .tag-pill {
    background: var(--panel);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .tag-pill.active { background: var(--info); color: #062129; border-color: var(--info); font-weight: 700; }
  button.btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 12px;
  }
  .split-layout {
    display: grid;
    grid-template-columns: minmax(340px, 1.15fr) minmax(360px, 1fr);
    gap: 14px;
    align-items: start;
  }
  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    max-height: 68vh;
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
  .badge.keep { border-color: var(--accent); color: var(--accent); }
  .badge.drop { border-color: var(--danger); color: var(--danger); }
  .badge.top1 { border-color: var(--warn); color: var(--warn); }
  .inspector {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .inspector img {
    width: 100%;
    max-height: 46vh;
    object-fit: contain;
    background: #05080c;
    border-radius: 6px;
    display: block;
  }
  .meter-track {
    width: 100%;
    height: 10px;
    background: var(--bg);
    border-radius: 999px;
    border: 1px solid var(--border);
    overflow: hidden;
    margin-top: 6px;
  }
  .meter-fill { height: 100%; background: var(--accent); }
  .kv-grid {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 6px 10px;
    margin-top: 12px;
    font-size: 13px;
  }
  .kv-key { color: var(--muted); }
  .pager { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Challenge 4: Edge Bandwidth Triage Distillation Explorer</h1>
  <div class="sub">5-Head Teacher Signals (<code>triage_keep</code>, <code>priority_score</code>, <code>modality</code>, <code>perspective</code>, <code>scene_tags</code>) &middot; 1% / 5% Uplink Budgets</div>
</header>

<div class="container">
  <div class="banner" id="gtHiddenBanner">Held-out <code>test</code> split: teacher labels are stripped in public archives so teams only score locally on <code>valid</code>.</div>
  <div class="stats-row" id="statsRow"></div>

  <div class="toolbar">
    <label>Split <select id="splitSelect"></select></label>
    <label>Triage / Budget
      <select id="keepFilter">
        <option value="all">All frames</option>
        <option value="keep_true">Keep = True</option>
        <option value="keep_false">Keep = False</option>
        <option value="high">High Priority (keep &amp; &gt;= 0.70)</option>
        <option value="top_1pct">Top 1% Budget</option>
        <option value="top_5pct">Top 5% Budget</option>
      </select>
    </label>
    <label>Modality
      <select id="modFilter">
        <option value="all">All</option>
        <option value="rgb">rgb</option>
        <option value="night">night</option>
        <option value="thermal">thermal</option>
      </select>
    </label>
    <label>Perspective
      <select id="perspFilter">
        <option value="all">All</option>
        <option value="from_drone">from_drone</option>
        <option value="ground">ground</option>
        <option value="of_drone">of_drone</option>
        <option value="other">other</option>
      </select>
    </label>
    <label>Sort
      <select id="sortSelect">
        <option value="prio_desc">Priority High -&gt; Low</option>
        <option value="prio_asc">Priority Low -&gt; High</option>
        <option value="pred_desc">Pred Score High -&gt; Low</option>
        <option value="id_asc">Frame ID (A-Z)</option>
      </select>
    </label>
    <label>Search <input id="searchBox" type="text" placeholder="c4_0001" /></label>
  </div>

  <div class="tag-bar" id="tagBar">
    <span style="font-size:12px;color:var(--muted);">Scene tag filter:</span>
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
      <img id="inspImg" alt="Selected frame" />
      <div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;font-size:13px;">
          <span id="inspTitle" style="font-weight:700;"></span>
          <span id="inspKeepBadge" class="badge"></span>
        </div>
        <div class="meter-track"><div class="meter-fill" id="inspPrioBar" style="width:0%"></div></div>
      </div>
      <div class="kv-grid" id="inspDetails"></div>
    </div>
  </div>
</div>

<script>
(() => {
  const PAGE_SIZE = 36;
  let state = null;
  let activeTags = new Set();
  let filteredItems = [];
  let page = 0;
  let selectedId = null;

  function el(id) { return document.getElementById(id); }

  function populateSplits() {
    const sel = el("splitSelect");
    sel.innerHTML = "";
    const order = ["valid", "train", "test", "calibration"];
    const keys = Object.keys(state.splits).sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    for (const k of keys) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${k} (${state.splits[k].counts.frame_count})`;
      sel.appendChild(opt);
    }
  }

  function renderTagBar() {
    const bar = el("tagBar");
    bar.innerHTML = "<span style='font-size:12px;color:#8b9bb0;'>Scene tags:</span>";
    for (const tag of state.scene_keys) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tag-pill" + (activeTags.has(tag) ? " active" : "");
      btn.textContent = tag;
      btn.onclick = () => {
        if (activeTags.has(tag)) activeTags.delete(tag);
        else activeTags.add(tag);
        renderTagBar();
        applyFilters();
      };
      bar.appendChild(btn);
    }
  }

  function mediaUrl(rel) {
    return "/media/" + String(rel).split("/").map(encodeURIComponent).join("/");
  }

  function renderStats(splitObj) {
    const c = splitObj.counts;
    const row = el("statsRow");
    row.innerHTML = "";
    const cards = [
      ["Frames / Clips", `${c.frame_count} / ${c.clip_count}`],
      ["Keep = True", splitObj.gt_available ? `${c.keep_true} (${Math.round(100 * c.keep_true / Math.max(1, c.frame_count))}%)` : "Hidden"],
      ["High Priority (>=0.7)", splitObj.gt_available ? c.high_priority : "Hidden"],
      ["1% / 5% Budget Slots", `${c.k_1pct} / ${c.k_5pct}`],
      ["Recall @ 1% Budget", c.recall_at_1pct != null ? c.recall_at_1pct.toFixed(4) : "N/A"],
      ["Recall @ 5% Budget", c.recall_at_5pct != null ? c.recall_at_5pct.toFixed(4) : "N/A"],
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

    const keepVal = el("keepFilter").value;
    const modVal = el("modFilter").value;
    const perspVal = el("perspFilter").value;
    const sortVal = el("sortSelect").value;
    const q = el("searchBox").value.trim().toLowerCase();

    filteredItems = splitObj.items.filter((it) => {
      if (q && !it.id.toLowerCase().includes(q) && !String(it.video_id || "").toLowerCase().includes(q)) return false;
      if (modVal !== "all" && it.modality !== modVal) return false;
      if (perspVal !== "all" && it.perspective !== perspVal) return false;
      if (keepVal === "keep_true" && it.triage_keep !== true) return false;
      if (keepVal === "keep_false" && it.triage_keep !== false) return false;
      if (keepVal === "high" && !(it.triage_keep === true && (it.priority_score || 0) >= 0.7)) return false;
      if (keepVal === "top_1pct" && it.budget_tier !== "top_1pct") return false;
      if (keepVal === "top_5pct" && it.budget_tier !== "top_1pct" && it.budget_tier !== "top_5pct") return false;
      for (const tag of activeTags) {
        if (!it.scene_tags || !it.scene_tags[tag]) return false;
      }
      return true;
    });

    filteredItems.sort((a, b) => {
      if (sortVal === "prio_desc") return (b.priority_score ?? -1) - (a.priority_score ?? -1);
      if (sortVal === "prio_asc") return (a.priority_score ?? 99) - (b.priority_score ?? 99);
      if (sortVal === "pred_desc") return (b.pred_priority ?? -1) - (a.pred_priority ?? -1);
      return a.id.localeCompare(b.id);
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
    el("pageInfo").textContent = `Page ${page + 1} / ${totalPages} (${filteredItems.length} frames)`;

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
      img.src = mediaUrl(it.media_src);
      img.alt = it.id;
      card.appendChild(img);

      const meta = document.createElement("div");
      meta.className = "thumb-meta";
      const s1 = document.createElement("span");
      s1.textContent = it.id;
      const b = document.createElement("span");
      if (!gtAvail || it.triage_keep === null) {
        b.className = "badge";
        b.textContent = it.pred_priority != null ? `P:${it.pred_priority.toFixed(2)}` : "hidden";
      } else {
        b.className = "badge " + (it.triage_keep ? (it.budget_tier === "top_1pct" ? "top1" : "keep") : "drop");
        b.textContent = `${it.triage_keep ? "KEEP" : "DROP"} ${(it.priority_score ?? 0).toFixed(2)}`;
      }
      meta.appendChild(s1);
      meta.appendChild(b);
      card.appendChild(meta);
      grid.appendChild(card);
    }

    const active = slice.find((x) => x.id === selectedId) || slice[0];
    if (active) renderInspector(active);
  }

  function renderInspector(it) {
    const splitObj = state.splits[el("splitSelect").value];
    const gtAvail = Boolean(splitObj && splitObj.gt_available);
    el("inspImg").src = mediaUrl(it.media_src);
    el("inspTitle").textContent = `${it.id} (clip ${it.video_id || "—"})`;

    const kb = el("inspKeepBadge");
    if (!gtAvail || it.triage_keep === null) {
      kb.className = "badge";
      kb.textContent = "Held-out / Calibration";
    } else {
      kb.className = "badge " + (it.triage_keep ? "keep" : "drop");
      kb.textContent = it.triage_keep ? "TEACHER KEEP" : "TEACHER DROP";
    }

    const scoreForBar = it.priority_score != null ? it.priority_score : (it.pred_priority ?? 0);
    el("inspPrioBar").style.width = `${Math.max(0, Math.min(100, scoreForBar * 100))}%`;

    const kv = el("inspDetails");
    kv.innerHTML = "";
    const addRow = (k, v) => {
      const kd = document.createElement("div");
      kd.className = "kv-key";
      kd.textContent = k;
      const vd = document.createElement("div");
      vd.textContent = v;
      kv.appendChild(kd);
      kv.appendChild(vd);
    };
    addRow("Resolution", `${it.width || "?"}x${it.height || "?"}`);
    addRow("Budget Tier", it.budget_tier || "—");
    addRow("Priority Score", it.priority_score != null ? it.priority_score.toFixed(4) : "Hidden");
    addRow("Modality", it.modality || "Hidden");
    addRow("Perspective", it.perspective || "Hidden");
    const activeList = it.scene_tags ? Object.keys(it.scene_tags).filter((k) => it.scene_tags[k]) : [];
    addRow("Scene Tags", activeList.length ? activeList.join(", ") : (it.scene_tags ? "none active" : "Hidden"));
    if (it.pred_priority != null || it.pred_keep != null) {
      addRow("Student Pred", `keep=${it.pred_keep} · score=${it.pred_priority != null ? it.pred_priority.toFixed(4) : "—"}`);
    }
  }

  async function init() {
    const resp = await fetch("/api/state");
    state = await resp.json();
    populateSplits();
    renderTagBar();
    applyFilters();

    for (const id of ["splitSelect", "keepFilter", "modFilter", "perspFilter", "sortSelect"]) {
      el(id).onchange = applyFilters;
    }
    el("searchBox").oninput = applyFilters;
    el("prevPage").onclick = () => { if (page > 0) { page--; renderPage(); } };
    el("nextPage").onclick = () => { page++; renderPage(); };
  }

  init();
})();
</script>
</body>
</html>
"""


def _load_triage_preds(preds_path: Path | None) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    if preds_path is None or not preds_path.is_file():
        return preds
    for line in preds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row["id"])
        prio_val: float | None = None
        if "priority_score" in row and row["priority_score"] is not None:
            try:
                parsed = float(row["priority_score"])
            except (TypeError, ValueError):
                parsed = float("nan")
            if math.isfinite(parsed):
                prio_val = parsed
        preds[item_id] = {
            "triage_keep": bool(row["triage_keep"])
            if isinstance(row.get("triage_keep"), bool)
            else None,
            "priority_score": prio_val,
        }
    return preds


def _compute_budget_tiers_and_recalls(
    items: list[dict[str, Any]], gt_available: bool
) -> tuple[int, int, float | None, float | None]:
    n = len(items)
    if n == 0:
        return 0, 0, None, None
    k_1pct = max(1, int(round(n * 0.01)))
    k_5pct = max(1, int(round(n * 0.05)))

    has_preds = all(it.get("pred_priority") is not None for it in items)
    if not has_preds and not gt_available:
        return k_1pct, k_5pct, None, None

    def _score_of(it: dict[str, Any]) -> float:
        if has_preds:
            return float(it["pred_priority"])
        return float(it.get("priority_score") or 0.0)

    def _is_high(it: dict[str, Any]) -> bool:
        return (
            bool(it.get("triage_keep"))
            and float(it.get("priority_score") or 0.0) >= 0.7
        )

    # Pessimistic tie-breaking: place non-positive frames first when scores tie
    ranked = sorted(
        items,
        key=lambda it: (
            -_score_of(it),
            1 if (gt_available and _is_high(it)) else 0,
            it["id"],
        ),
    )
    for idx, it in enumerate(ranked):
        if idx < k_1pct:
            it["budget_tier"] = "top_1pct"
        elif idx < k_5pct:
            it["budget_tier"] = "top_5pct"
        else:
            it["budget_tier"] = "rest"

    if not (gt_available and has_preds):
        return k_1pct, k_5pct, None, None

    positives = sum(1 for it in items if _is_high(it))
    hits_1 = sum(1 for it in ranked[:k_1pct] if _is_high(it))
    hits_5 = sum(1 for it in ranked[:k_5pct] if _is_high(it))
    r1 = round(hits_1 / max(1, min(k_1pct, positives)), 4)
    r5 = round(hits_5 / max(1, min(k_5pct, positives)), 4)
    return k_1pct, k_5pct, r1, r5


def build_state(
    challenge_dir: Path,
    *,
    preds_path: Path | None = None,
) -> dict[str, Any]:
    """Index Challenge 4 splits, multi-head VLM labels, calibration, and uplink budgets."""
    dataset_dir = challenge_dir / "dataset"
    preds_by_id = _load_triage_preds(preds_path)
    splits: dict[str, Any] = {}
    train_meta_by_id: dict[str, dict[str, Any]] = {}

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
            "triage_keep" in row and "priority_score" in row for row in raw_rows
        )
        items: list[dict[str, Any]] = []
        clips: set[str] = set()
        keep_true = 0
        high_prio = 0
        for row in raw_rows:
            item_id = str(row["id"])
            vid = str(row.get("video_id", ""))
            if vid:
                clips.add(vid)
            keep_val = bool(row["triage_keep"]) if "triage_keep" in row else None
            prio_val = (
                float(row["priority_score"])
                if "priority_score" in row
                and math.isfinite(float(row["priority_score"]))
                else None
            )
            if keep_val is True:
                keep_true += 1
                if prio_val is not None and prio_val >= 0.7:
                    high_prio += 1
            pred = preds_by_id.get(item_id, {})
            item_obj = {
                "id": item_id,
                "video_id": vid or None,
                "split": split,
                "media_src": f"dataset/{row['image']}",
                "width": int(row.get("width", 0)),
                "height": int(row.get("height", 0)),
                "triage_keep": keep_val,
                "priority_score": prio_val,
                "modality": row.get("modality"),
                "perspective": row.get("perspective"),
                "scene_tags": row.get("scene_tags"),
                "pred_keep": pred.get("triage_keep"),
                "pred_priority": pred.get("priority_score"),
                "budget_tier": None,
            }
            if split == "train":
                train_meta_by_id[item_id] = item_obj
            items.append(item_obj)

        k_1pct, k_5pct, r1, r5 = _compute_budget_tiers_and_recalls(items, gt_available)
        splits[split] = {
            "gt_available": gt_available,
            "counts": {
                "frame_count": len(items),
                "clip_count": len(clips),
                "keep_true": keep_true,
                "keep_false": len(items) - keep_true if gt_available else 0,
                "high_priority": high_prio,
                "k_1pct": k_1pct,
                "k_5pct": k_5pct,
                "recall_at_1pct": r1,
                "recall_at_5pct": r5,
            },
            "items": items,
        }

    cal_dir = challenge_dir / "calibration"
    if cal_dir.is_dir():
        cal_files = sorted(
            p
            for p in cal_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        cal_items: list[dict[str, Any]] = []
        for p in cal_files:
            ref = train_meta_by_id.get(p.stem, {})
            cal_items.append(
                {
                    "id": p.stem,
                    "video_id": ref.get("video_id"),
                    "split": "calibration",
                    "media_src": f"calibration/{p.name}",
                    "width": ref.get("width", 0),
                    "height": ref.get("height", 0),
                    "triage_keep": None,
                    "priority_score": None,
                    "modality": ref.get("modality"),
                    "perspective": ref.get("perspective"),
                    "scene_tags": None,
                    "pred_keep": None,
                    "pred_priority": None,
                    "budget_tier": None,
                }
            )
        splits["calibration"] = {
            "gt_available": False,
            "counts": {
                "frame_count": len(cal_items),
                "clip_count": len(
                    {it["video_id"] for it in cal_items if it.get("video_id")}
                ),
                "keep_true": 0,
                "keep_false": 0,
                "high_priority": 0,
                "k_1pct": 0,
                "k_5pct": 0,
                "recall_at_1pct": None,
                "recall_at_5pct": None,
            },
            "items": cal_items,
        }

    return {
        "challenge": "challenge-4-edge-triage",
        "title": "Challenge 4: Edge Bandwidth Triage Distillation",
        "scene_keys": list(SCENE_KEYS),
        "modalities": list(MODALITIES),
        "perspectives": list(PERSPECTIVES),
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
    port: int = 8767,
    preds_path: Path | None = None,
) -> ThreadingHTTPServer:
    """Create the Challenge 4 HTTP viewer server bound to ``host:port``."""
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
        help="Challenge 4 directory containing dataset/ and calibration/",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--preds",
        type=Path,
        default=None,
        help="Optional triage_preds.jsonl to score top-1%%/5%% bandwidth capture",
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
    print(f"Challenge 4 viewer listening on {actual_url}")
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
