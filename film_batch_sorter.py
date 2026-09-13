#!/usr/bin/env python3
"""
Film Batch Sorter — local web app
Supports two vision backends:
  • Ollama  (local, free, no internet — recommended)
  • Anthropic Claude (cloud API, requires key + tokens)
"""

import os, re, json, sys, shutil, base64, queue, threading, webbrowser, subprocess
import urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── auto-install ──────────────────────────────────────────────────────────────
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

try:
    from flask import Flask, request, jsonify, Response, stream_with_context
except ImportError:
    print("Installing flask…"); _pip("flask")
    from flask import Flask, request, jsonify, Response, stream_with_context

try:
    import anthropic
except ImportError:
    print("Installing anthropic…"); _pip("anthropic")
    import anthropic

try:
    from PIL import Image, ImageOps
    import io
except ImportError:
    print("Installing pillow…"); _pip("pillow")
    from PIL import Image, ImageOps
    import io

try:
    import rawpy
except ImportError:
    print("Installing rawpy…"); _pip("rawpy")
    import rawpy

SUPPORTED_EXT = {".jpg", ".jpeg", ".nef"}
OLLAMA_URL    = "http://localhost:11434"
PARALLEL_WORKERS_CLOUD = 6   # concurrent calls for Anthropic
PARALLEL_WORKERS_LOCAL = 1   # Ollama processes one at a time on GPU

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path.home() / ".film_batch_sorter.json"
PORT = 5174

def load_cfg():
    try: return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    except: return {}

def save_cfg(d):
    try: CONFIG_PATH.write_text(json.dumps(d))
    except: pass

# ── flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
_jobs: dict[str, queue.Queue] = {}

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "GET":
        return jsonify(load_cfg())
    save_cfg(request.json or {})
    return jsonify({"ok": True})

@app.route("/pick-folder", methods=["POST"])
def pick_folder():
    script = 'return POSIX path of (choose folder with prompt "Select folder:")'
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    path = r.stdout.strip() if r.returncode == 0 else ""
    return jsonify({"path": path})

@app.route("/check-ollama", methods=["GET"])
def check_ollama():
    """Check if Ollama is running and return available vision models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        # Filter to known vision-capable models
        vision_models = {"llava", "llava-phi3", "llava:13b", "llava:34b",
                         "moondream", "bakllava", "minicpm-v", "llava-llama3"}
        models = [m["name"] for m in data.get("models", [])
                  if any(v in m["name"].lower() for v in vision_models)]
        return jsonify({"running": True, "models": models})
    except Exception:
        return jsonify({"running": False, "models": []})

@app.route("/scan", methods=["POST"])
def scan():
    data = request.json or {}
    jid = "job"
    q: queue.Queue = queue.Queue()
    _jobs[jid] = q
    threading.Thread(target=_run_scan, args=(data, q), daemon=True).start()
    return jsonify({"job_id": jid})

@app.route("/progress/<jid>")
def progress(jid):
    q = _jobs.get(jid)
    if not q:
        return "Not found", 404
    def generate():
        while True:
            try:
                msg = q.get(timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("done"):
                    break
            except queue.Empty:
                yield "data: {}\n\n"
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── image prep (shared) ───────────────────────────────────────────────────────
def _load_rotations(path: Path) -> list[str]:
    """Load image, resize, return list of base64 JPEG strings at 0° and 180°."""
    if path.suffix.lower() == ".nef":
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        import numpy as np
        img = Image.fromarray(rgb)
    else:
        img = Image.open(path).convert("RGB")

    max_side = 1024
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    b64_list = []
    for angle in (0, 180):
        rotated = img.rotate(angle, expand=True)
        buf = io.BytesIO()
        rotated.save(buf, format="JPEG", quality=82)
        b64_list.append(base64.b64encode(buf.getvalue()).decode())
    return b64_list

PROMPT = (
    "I am sending you the same image at 0° and 180° rotation. "
    "Look for a small rectangular label that contains a printed "
    "(never handwritten) 3-digit number — it may look like a sticker, "
    "paper tag, or printed strip. "
    "One of the two rotations will show it the right way up. "
    "Reply with ONLY the 3-digit number exactly as printed, e.g. \"042\". "
    "If no such printed 3-digit label is visible in any rotation, "
    "reply with only \"NO\"."
)

# ── Anthropic backend ─────────────────────────────────────────────────────────
def _vision_call_anthropic(client: anthropic.Anthropic, path: Path) -> str:
    b64_list = _load_rotations(path)
    content = [
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/jpeg", "data": b}}
        for b in b64_list
    ]
    content.append({"type": "text", "text": PROMPT})
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=32,
        messages=[{"role": "user", "content": content}]
    )
    return resp.content[0].text.strip()

# ── Ollama backend ────────────────────────────────────────────────────────────
def _vision_call_ollama(model: str, path: Path) -> str:
    b64_list = _load_rotations(path)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": b64_list}],
        "stream": False,
        "options": {"temperature": 0}
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()

# ── scan worker ───────────────────────────────────────────────────────────────
def _nat_key(p: Path):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)]

def _run_scan(data: dict, q: queue.Queue):
    def send(**kw): q.put(kw)

    backend      = data.get("backend", "anthropic")   # "anthropic" | "ollama"
    api_key      = data.get("api_key", "").strip()
    ollama_model = data.get("ollama_model", "llava-phi3").strip()
    inp_dir      = Path(data.get("input_dir", ""))
    out_dir      = Path(data.get("output_dir", ""))
    recursive    = data.get("recursive", False)

    # Validate
    if backend == "anthropic" and not api_key:
        return send(tag="err", msg="No Anthropic API key provided.", done=True)
    if not inp_dir.is_dir():
        return send(tag="err", msg=f"Input folder not found: {inp_dir}", done=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    if recursive:
        files = sorted(
            {p for p in inp_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXT and p.is_file()},
            key=_nat_key
        )
    else:
        files = sorted(
            [p for p in inp_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT and p.is_file()],
            key=_nat_key
        )

    total = len(files)
    if not total:
        return send(tag="info", msg="No images found in the selected folder.", done=True)

    workers = PARALLEL_WORKERS_LOCAL if backend == "ollama" else PARALLEL_WORKERS_CLOUD
    backend_label = f"Ollama ({ollama_model})" if backend == "ollama" else "Anthropic Claude"
    send(tag="info",
         msg=f"Found {total} image(s) — scanning with {backend_label}, {workers} at a time…",
         total=total, progress=0)

    # Set up backend client
    client = anthropic.Anthropic(api_key=api_key) if backend == "anthropic" else None

    # ── Phase 1: parallel vision scan ─────────────────────────────────────────
    answers = [None] * total
    done_count = [0]
    lock = threading.Lock()

    def scan_one(idx: int, path: Path):
        try:
            if backend == "ollama":
                answer = _vision_call_ollama(ollama_model, path)
            else:
                answer = _vision_call_anthropic(client, path)
            result = ("OK", answer)
        except Exception as exc:
            result = ("ERR", str(exc))
        answers[idx] = result
        with lock:
            done_count[0] += 1
            cnt = done_count[0]
        send(progress=round((cnt / total) * 80), current=f"{cnt}/{total} scanned")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one, i, p): i for i, p in enumerate(files)}
        for _ in as_completed(futures):
            pass

    # ── Phase 2: assign batches and copy ──────────────────────────────────────
    send(tag="info", msg="Scan complete — assigning batches and copying files…")
    current_folder = None
    n_batches = n_copied = n_skipped = n_errors = 0

    for i, path in enumerate(files):
        status, value = answers[i]
        pct = 80 + round((i / total) * 20)

        if status == "ERR":
            send(tag="err", msg=f"  ERROR  {path.name}: {value}", progress=pct)
            n_errors += 1
            continue

        answer = value
        if re.match(r"^\d{3}$", answer):
            folder_name = f"Film{answer}"
            current_folder = out_dir / folder_name
            current_folder.mkdir(parents=True, exist_ok=True)
            n_batches += 1
            send(tag="batch", msg=f"★  {path.name}  →  [{folder_name}/]  (new batch)", progress=pct)
        elif current_folder:
            send(tag="ok", msg=f"   {path.name}  →  {current_folder.name}/", progress=pct)
        else:
            send(tag="skip", msg=f"   {path.name}  →  (skipped — no batch started yet)", progress=pct)
            n_skipped += 1
            continue

        try:
            shutil.copy2(path, current_folder / path.name)
            n_copied += 1
        except Exception as exc:
            send(tag="err", msg=f"  COPY ERROR  {path.name}: {exc}")
            n_errors += 1

    send(
        tag="done",
        msg=f"\n✓  Complete — {n_batches} batch(es) · {n_copied} copied · {n_skipped} skipped · {n_errors} errors",
        progress=100, done=True,
        summary={"batches": n_batches, "copied": n_copied, "skipped": n_skipped, "errors": n_errors}
    )

# ── embedded HTML ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Film Batch Sorter</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f9f8f5; --surface: #ffffff; --border: #ddd9ce;
    --text: #1a1916; --muted: #777068; --amber: #d48a0a;
    --amber-bg: #fdf3dc; --green: #2d6e0e; --green-bg: #eaf3de;
    --red: #b03030; --red-bg: #fdecec; --blue: #1055a0; --blue-bg: #e6f1fb;
    --radius: 10px; --mono: "Menlo", "Consolas", monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #1a1916; --surface: #242320; --border: #3a3830;
            --text: #e8e4d8; --muted: #8a8470; --amber: #e8a820;
            --amber-bg: #2e2408; --green: #5aa82a; --green-bg: #142808;
            --red: #e05050; --red-bg: #280a0a; --blue: #5090e0; --blue-bg: #081428; }
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 14px; line-height: 1.5; min-height: 100vh; display: flex; flex-direction: column; }
  .topbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; gap: 10px; }
  .topbar h1 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .topbar .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--amber); flex-shrink: 0; }
  .main { flex: 1; padding: 24px; max-width: 820px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; }
  .card-title { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
  .field { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .field:last-child { margin-bottom: 0; }
  .field label { width: 100px; font-size: 13px; color: var(--muted); flex-shrink: 0; }
  .field-row { flex: 1; display: flex; gap: 8px; }
  input[type=text], input[type=password], select {
    flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 7px;
    padding: 7px 11px; font-size: 13px; color: var(--text); outline: none; transition: border-color .15s; font-family: inherit;
  }
  input[type=text]:focus, input[type=password]:focus, select:focus { border-color: var(--amber); }
  .btn { cursor: pointer; border-radius: 7px; font-size: 13px; font-weight: 500; padding: 7px 14px; border: 1px solid var(--border); background: var(--bg); color: var(--text); transition: background .12s, border-color .12s; white-space: nowrap; font-family: inherit; }
  .btn:hover { background: var(--border); }
  .btn-primary { background: var(--amber); color: #fff; border-color: var(--amber); font-size: 14px; padding: 9px 22px; }
  .btn-primary:hover { opacity: .88; }
  .btn-primary:disabled { opacity: .45; cursor: not-allowed; }
  .btn-sm { padding: 5px 10px; font-size: 12px; }
  .check-row { display: flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); cursor: pointer; }
  .check-row input { accent-color: var(--amber); width: 15px; height: 15px; }
  .progress-wrap { background: var(--bg); border-radius: 4px; height: 4px; overflow: hidden; margin: 4px 0 12px; }
  .progress-bar { height: 100%; background: var(--amber); transition: width .4s ease; border-radius: 4px; width: 0%; }
  .log { background: var(--bg); border-radius: 8px; padding: 12px 14px; height: 300px; overflow-y: auto; font-family: var(--mono); font-size: 12px; line-height: 1.8; border: 1px solid var(--border); }
  .log-line { display: block; white-space: pre-wrap; word-break: break-all; }
  .t-batch { color: var(--amber); font-weight: 600; }
  .t-ok    { color: var(--green); }
  .t-skip  { color: var(--muted); }
  .t-err   { color: var(--red); }
  .t-info  { color: var(--blue); }
  .t-done  { color: var(--blue); font-weight: 600; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
  .stat { background: var(--bg); border-radius: 8px; padding: 12px 14px; border: 1px solid var(--border); }
  .stat-val { font-size: 22px; font-weight: 600; color: var(--text); }
  .stat-lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .status-bar { font-size: 12px; color: var(--muted); margin-bottom: 8px; min-height: 18px; }
  .actions { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
  .hidden { display: none !important; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 500; }
  .badge-green { background: var(--green-bg); color: var(--green); }
  .badge-red   { background: var(--red-bg);   color: var(--red); }
  .badge-muted { background: var(--bg); color: var(--muted); border: 1px solid var(--border); }
  .segment { display: flex; border: 1px solid var(--border); border-radius: 7px; overflow: hidden; }
  .segment button { flex: 1; padding: 7px 14px; font-size: 13px; font-weight: 500; border: none; background: var(--bg); color: var(--muted); cursor: pointer; transition: background .12s, color .12s; font-family: inherit; }
  .segment button.active { background: var(--amber); color: #fff; }
</style>
</head>
<body>

<div class="topbar">
  <div class="dot"></div>
  <h1>Film Batch Sorter</h1>
</div>

<div class="main">

  <!-- Backend + Settings -->
  <div class="card">
    <div class="card-title">Vision Backend</div>

    <div class="field">
      <label>Backend</label>
      <div class="segment" id="backendSeg">
        <button class="active" onclick="setBackend('ollama')">🖥 Local (Ollama)</button>
        <button onclick="setBackend('anthropic')">☁️ Anthropic Claude</button>
      </div>
    </div>

    <!-- Ollama fields -->
    <div id="ollamaFields">
      <div class="field">
        <label>Model</label>
        <div class="field-row">
          <select id="ollamaModel">
            <option value="llava-phi3">llava-phi3 (recommended, ~2.9 GB)</option>
            <option value="llava">llava (more capable, ~4.7 GB)</option>
            <option value="moondream">moondream (fastest, ~1.7 GB)</option>
            <option value="minicpm-v">minicpm-v (~5.5 GB)</option>
          </select>
          <button class="btn btn-sm" onclick="checkOllama()">Check</button>
        </div>
      </div>
      <div class="field">
        <label></label>
        <span id="ollamaStatus" class="badge badge-muted">Not checked</span>
      </div>
    </div>

    <!-- Anthropic fields -->
    <div id="anthropicFields" class="hidden">
      <div class="field">
        <label>API key</label>
        <div class="field-row">
          <input type="password" id="apiKey" placeholder="sk-ant-…" autocomplete="off">
          <button class="btn btn-sm" onclick="toggleKey()">Show</button>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Folders</div>

    <div class="field">
      <label>Input folder</label>
      <div class="field-row">
        <input type="text" id="inputDir" placeholder="Path to incoming images folder…" readonly>
        <button class="btn btn-sm" onclick="pickFolder('inputDir')">Browse…</button>
      </div>
    </div>

    <div class="field">
      <label>Output folder</label>
      <div class="field-row">
        <input type="text" id="outputDir" placeholder="Where to create Film### subfolders…" readonly>
        <button class="btn btn-sm" onclick="pickFolder('outputDir')">Browse…</button>
      </div>
    </div>

    <div class="field" style="margin-top:6px;">
      <label></label>
      <label class="check-row"><input type="checkbox" id="recursive"> Scan subfolders recursively</label>
    </div>
  </div>

  <!-- Progress -->
  <div class="card" id="progressCard" style="display:none;">
    <div class="card-title">Progress</div>
    <div class="status-bar" id="statusBar">Ready</div>
    <div class="progress-wrap"><div class="progress-bar" id="progBar"></div></div>
    <div class="log" id="logBox"></div>
    <div class="stats hidden" id="statsBox">
      <div class="stat"><div class="stat-val" id="sBatches">—</div><div class="stat-lbl">batches</div></div>
      <div class="stat"><div class="stat-val" id="sCopied">—</div><div class="stat-lbl">copied</div></div>
      <div class="stat"><div class="stat-val" id="sSkipped">—</div><div class="stat-lbl">skipped</div></div>
      <div class="stat"><div class="stat-val" id="sErrors">—</div><div class="stat-lbl">errors</div></div>
    </div>
  </div>

  <!-- Actions -->
  <div class="actions">
    <div style="font-size:12px; color:var(--muted);">Settings saved automatically</div>
    <div style="display:flex; gap:10px;">
      <button class="btn" id="cancelBtn" onclick="cancelScan()" style="display:none;">Cancel</button>
      <button class="btn btn-primary" id="runBtn" onclick="startScan()">Sort batches ▶</button>
    </div>
  </div>

</div>

<script>
let evtSource = null;
let currentBackend = 'ollama';

// ── load config ───────────────────────────────────────────────────────────────
fetch('/config').then(r => r.json()).then(cfg => {
  if (cfg.api_key)      document.getElementById('apiKey').value    = cfg.api_key;
  if (cfg.input_dir)    document.getElementById('inputDir').value  = cfg.input_dir;
  if (cfg.output_dir)   document.getElementById('outputDir').value = cfg.output_dir;
  if (cfg.recursive)    document.getElementById('recursive').checked = cfg.recursive;
  if (cfg.ollama_model) document.getElementById('ollamaModel').value = cfg.ollama_model;
  if (cfg.backend)      setBackend(cfg.backend, false);
}).catch(() => {});

// Auto-check Ollama on load
window.addEventListener('load', () => { if (currentBackend === 'ollama') checkOllama(); });

function setBackend(b, save = true) {
  currentBackend = b;
  const btns = document.querySelectorAll('#backendSeg button');
  btns[0].classList.toggle('active', b === 'ollama');
  btns[1].classList.toggle('active', b === 'anthropic');
  document.getElementById('ollamaFields').classList.toggle('hidden', b !== 'ollama');
  document.getElementById('anthropicFields').classList.toggle('hidden', b !== 'anthropic');
  if (save) saveConfig();
}

function saveConfig() {
  const cfg = {
    backend:      currentBackend,
    api_key:      document.getElementById('apiKey').value.trim(),
    ollama_model: document.getElementById('ollamaModel').value,
    input_dir:    document.getElementById('inputDir').value.trim(),
    output_dir:   document.getElementById('outputDir').value.trim(),
    recursive:    document.getElementById('recursive').checked,
  };
  fetch('/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(cfg) });
}

function checkOllama() {
  const el = document.getElementById('ollamaStatus');
  el.className = 'badge badge-muted'; el.textContent = 'Checking…';
  fetch('/check-ollama').then(r => r.json()).then(d => {
    if (d.running) {
      const sel = document.getElementById('ollamaModel');
      const cur = sel.value;
      if (d.models.length) {
        // Add any installed models not already in the list
        d.models.forEach(m => {
          if (![...sel.options].some(o => o.value === m)) {
            const opt = document.createElement('option'); opt.value = m; opt.textContent = m;
            sel.insertBefore(opt, sel.firstChild);
          }
        });
        // Prefer an installed model
        const installed = d.models.find(m => sel.value === m) || d.models[0];
        sel.value = installed || cur;
        el.className = 'badge badge-green';
        el.textContent = `✓ Ollama running — ${d.models.length} vision model(s) installed`;
      } else {
        el.className = 'badge badge-red';
        el.textContent = '⚠ Ollama running but no vision model found — see setup below';
      }
    } else {
      el.className = 'badge badge-red';
      el.textContent = '✗ Ollama not running — install from ollama.com';
    }
  }).catch(() => {
    el.className = 'badge badge-red';
    el.textContent = '✗ Could not reach Ollama';
  });
}

function toggleKey() {
  const el = document.getElementById('apiKey');
  el.type = el.type === 'password' ? 'text' : 'password';
}

function pickFolder(fieldId) {
  fetch('/pick-folder', {method: 'POST'})
    .then(r => r.json())
    .then(d => { if (d.path) { document.getElementById(fieldId).value = d.path; saveConfig(); } });
}

function log(msg, tag) {
  const box = document.getElementById('logBox');
  const el = document.createElement('span');
  el.className = 'log-line t-' + (tag || 'info');
  el.textContent = msg;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

function startScan() {
  const inputDir  = document.getElementById('inputDir').value.trim();
  const outputDir = document.getElementById('outputDir').value.trim();
  const apiKey    = document.getElementById('apiKey').value.trim();

  if (!inputDir)  { alert('Please select an input folder.');   return; }
  if (!outputDir) { alert('Please select an output folder.');  return; }
  if (currentBackend === 'anthropic' && !apiKey) {
    alert('Please enter your Anthropic API key.'); return;
  }

  saveConfig();
  document.getElementById('logBox').innerHTML = '';
  document.getElementById('statsBox').classList.add('hidden');
  document.getElementById('progressCard').style.display = '';
  document.getElementById('progBar').style.width = '0%';
  document.getElementById('statusBar').textContent = 'Starting…';
  document.getElementById('runBtn').disabled = true;
  document.getElementById('cancelBtn').style.display = '';

  fetch('/scan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      backend:      currentBackend,
      api_key:      apiKey,
      ollama_model: document.getElementById('ollamaModel').value,
      input_dir:    inputDir,
      output_dir:   outputDir,
      recursive:    document.getElementById('recursive').checked
    })
  })
  .then(r => r.json())
  .then(d => {
    evtSource = new EventSource('/progress/' + d.job_id);
    evtSource.onmessage = function(e) {
      if (!e.data || e.data === '{}') return;
      const msg = JSON.parse(e.data);
      if (msg.msg)      log(msg.msg, msg.tag);
      if (msg.progress !== undefined) document.getElementById('progBar').style.width = msg.progress + '%';
      if (msg.current)  document.getElementById('statusBar').textContent = 'Scanning: ' + msg.current;
      if (msg.done) {
        evtSource.close(); evtSource = null;
        document.getElementById('runBtn').disabled = false;
        document.getElementById('cancelBtn').style.display = 'none';
        document.getElementById('statusBar').textContent = 'Done';
        if (msg.summary) {
          const s = msg.summary;
          document.getElementById('sBatches').textContent = s.batches;
          document.getElementById('sCopied').textContent  = s.copied;
          document.getElementById('sSkipped').textContent = s.skipped;
          document.getElementById('sErrors').textContent  = s.errors;
          document.getElementById('statsBox').classList.remove('hidden');
        }
      }
    };
    evtSource.onerror = function() {
      log('Connection error — check Terminal for details.', 'err');
      document.getElementById('runBtn').disabled = false;
      document.getElementById('cancelBtn').style.display = 'none';
    };
  });
}

function cancelScan() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  log('\nCancelled.', 'err');
  document.getElementById('runBtn').disabled = false;
  document.getElementById('cancelBtn').style.display = 'none';
  document.getElementById('statusBar').textContent = 'Cancelled';
}
</script>
</body>
</html>
"""

# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"\n  Film Batch Sorter  →  {url}\n  Press Ctrl-C to quit.\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(port=PORT, threaded=True, debug=False)
