#!/usr/bin/env python3
"""
Film Batch Sorter — local web app
Installs flask + anthropic if needed, then opens http://localhost:5174
"""

import os, re, json, sys, shutil, base64, queue, threading, webbrowser, subprocess
from pathlib import Path

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

# ── helpers ───────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed

PARALLEL_WORKERS = 6   # concurrent API calls

def _nat_key(p: Path):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)]

def _vision_call(client: anthropic.Anthropic, path: Path) -> str:
    """Return 3-digit string if a batch marker is found, otherwise 'NO'."""
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

    content = []
    for angle in (0, 180):
        rotated = img.rotate(angle, expand=True)
        buf = io.BytesIO()
        rotated.save(buf, format="JPEG", quality=82)
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode()
        }})
    content.append({"type": "text", "text": (
        "I am sending you the same image at 0° and 180° rotation. "
        "Look for a small rectangular label that contains a printed "
        "(never handwritten) 3-digit number — it may look like a sticker, "
        "paper tag, or printed strip. "
        "One of the two rotations will show it the right way up. "
        "Reply with ONLY the 3-digit number exactly as printed, e.g. \"042\". "
        "If no such printed 3-digit label is visible in any rotation, "
        "reply with only \"NO\"."
    )})

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=32,
        messages=[{"role": "user", "content": content}]
    )
    return resp.content[0].text.strip()

def _run_scan(data: dict, q: queue.Queue):
    def send(**kw): q.put(kw)

    api_key   = data.get("api_key", "").strip()
    inp_dir   = Path(data.get("input_dir", ""))
    out_dir   = Path(data.get("output_dir", ""))
    recursive = data.get("recursive", False)

    if not api_key:
        return send(tag="err", msg="No API key provided.", done=True)
    if not inp_dir.is_dir():
        return send(tag="err", msg=f"Input folder not found: {inp_dir}", done=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        jpgs = sorted(
            {p for p in inp_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXT and p.is_file()},
            key=_nat_key
        )
    else:
        jpgs = sorted(
            [p for p in inp_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT and p.is_file()],
            key=_nat_key
        )

    total = len(jpgs)
    if not total:
        return send(tag="info", msg="No images found in the selected folder.", done=True)

    send(tag="info", msg=f"Found {total} image(s) — scanning {PARALLEL_WORKERS} at a time…",
         total=total, progress=0)

    client = anthropic.Anthropic(api_key=api_key)

    # ── Phase 1: parallel vision scan ─────────────────────────────────────────
    # answers[i] = ("OK", answer_str) | ("ERR", error_str)
    answers = [None] * total
    completed = threading.Semaphore(0)   # counts finished tasks
    done_count = [0]
    lock = threading.Lock()

    def scan_one(idx: int, path: Path):
        try:
            answer = _vision_call(client, path)
            result = ("OK", answer)
        except Exception as exc:
            result = ("ERR", str(exc))
        answers[idx] = result
        with lock:
            done_count[0] += 1
            cnt = done_count[0]
        send(progress=round((cnt / total) * 80),   # 0-80% = scanning phase
             current=f"{cnt}/{total} scanned")

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(scan_one, i, p): i for i, p in enumerate(jpgs)}
        for _ in as_completed(futures):
            pass   # progress sent from inside scan_one

    # ── Phase 2: assign batches and copy (sequential — order matters) ─────────
    send(tag="info", msg="Scan complete — assigning batches and copying files…")
    current_folder = None
    n_batches = n_copied = n_skipped = n_errors = 0

    for i, path in enumerate(jpgs):
        status, value = answers[i]
        pct = 80 + round((i / total) * 20)          # 80-100% = copy phase

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
  input[type=text], input[type=password] {
    flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 7px;
    padding: 7px 11px; font-size: 13px; color: var(--text); outline: none; transition: border-color .15s; font-family: inherit;
  }
  input[type=text]:focus, input[type=password]:focus { border-color: var(--amber); }
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
</style>
</head>
<body>

<div class="topbar">
  <div class="dot"></div>
  <h1>Film Batch Sorter</h1>
</div>

<div class="main">

  <!-- Settings -->
  <div class="card">
    <div class="card-title">Configuration</div>

    <div class="field">
      <label>API key</label>
      <div class="field-row">
        <input type="password" id="apiKey" placeholder="sk-ant-…" autocomplete="off">
        <button class="btn btn-sm" onclick="toggleKey()">Show</button>
      </div>
    </div>

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

// ── load config ───────────────────────────────────────────────────────────────
fetch('/config').then(r => r.json()).then(cfg => {
  if (cfg.api_key)    document.getElementById('apiKey').value    = cfg.api_key;
  if (cfg.input_dir)  document.getElementById('inputDir').value  = cfg.input_dir;
  if (cfg.output_dir) document.getElementById('outputDir').value = cfg.output_dir;
  if (cfg.recursive)  document.getElementById('recursive').checked = cfg.recursive;
}).catch(() => {});

function saveConfig() {
  const cfg = {
    api_key:    document.getElementById('apiKey').value.trim(),
    input_dir:  document.getElementById('inputDir').value.trim(),
    output_dir: document.getElementById('outputDir').value.trim(),
    recursive:  document.getElementById('recursive').checked,
  };
  fetch('/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(cfg) });
}

// ── UI helpers ────────────────────────────────────────────────────────────────
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

// ── scan ──────────────────────────────────────────────────────────────────────
function startScan() {
  const apiKey   = document.getElementById('apiKey').value.trim();
  const inputDir = document.getElementById('inputDir').value.trim();
  const outputDir= document.getElementById('outputDir').value.trim();

  if (!apiKey)    { alert('Please enter your Anthropic API key.'); return; }
  if (!inputDir)  { alert('Please select an input folder.');        return; }
  if (!outputDir) { alert('Please select an output folder.');       return; }

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
      api_key: apiKey, input_dir: inputDir, output_dir: outputDir,
      recursive: document.getElementById('recursive').checked
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
        evtSource.close();
        evtSource = null;
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
