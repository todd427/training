#!/usr/bin/env python3
# jsonl_ui.py — tiny JSONL browser/editor with role colors + PII highlighting

import os, re, json, shutil, html as _html
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def _discover_data_dir() -> Path:
    env = os.getenv("DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    for guess in (Path("data/sft_ready"), Path(".")):
        if guess.exists():
            return guess.resolve()
    p = Path(".").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

DATA_DIR: Path = _discover_data_dir()
APP_TITLE = "JSONL Editor"
PAGE_SIZE_DEFAULT = 20
PORT = int(os.getenv("PORT", "8002"))

# Simple PII detectors for preview highlighting
PII_PATTERNS = {
    "url": r"\bhttps?://\S+|www\.\S+",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "handle": r"(?<!\w)@([A-Za-z0-9_]{2,})",
    "phone": r"(?:(?:\+?\d{1,3}[ \-]?)?(?:\(?\d{2,4}\)?[ \-]?)?\d{3,4}[ \-]?\d{4})",
}

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = FastAPI(title=APP_TITLE)
BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "templates"))
static_dir = BASE / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def list_jsonl_files() -> List[Path]:
    # gather unique *.jsonl files from DATA_DIR; ignore backups
    files = []
    seen = set()
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        if p.suffixes[-1] == ".jsonl" and not str(p).endswith(".jsonl.bak"):
            if p.name not in seen:
                seen.add(p.name)
                files.append(p)
    return files

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                rows.append({"__error__": f"Line {i}: {e}", "__raw__": s})
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    # atomic-ish save with backup
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)

def highlight_pii(text: str) -> str:
    html = text
    for label, pat in PII_PATTERNS.items():
        html = re.sub(pat, lambda m: f'<mark class="pii" title="{label}">{m.group(0)}</mark>',
                      html, flags=re.I | re.M)
    return html

ROLE_CLASS = {
    "system": "role-system",
    "user": "role-user",
    "assistant": "role-assistant",
}

def render_messages_html(msgs: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for m in msgs:
        role = (m.get("role") or "").lower()
        cls = ROLE_CLASS.get(role, "role-unknown")
        content = str(m.get("content", ""))
        safe = _html.escape(content)
        safe = highlight_pii(safe)
        parts.append(
            f'<div class="bubble {cls}">'
            f'  <div class="bubble-head">{_html.escape(role or "unknown")}</div>'
            f'  <div class="bubble-body">{safe}</div>'
            f'</div>'
        )
    return "\n".join(parts)

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    files = [p.name for p in list_jsonl_files()]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "files": files,
        "data_dir": str(DATA_DIR),
    })

@app.get("/file/{name}", response_class=HTMLResponse)
def view_file(name: str, request: Request, page: int = 1, size: int = PAGE_SIZE_DEFAULT,
              q: str = "", regex: bool = False):
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(404, "File not found")
    rows = read_jsonl(path)

    # Filter
    if q:
        def _match(obj):
            s = json.dumps(obj, ensure_ascii=False)
            return (re.search(q, s, re.I | re.M) is not None) if regex else (q.lower() in s.lower())
        rows = [r for r in rows if _match(r)]

    total = len(rows)
    page = max(1, page)
    size = max(1, size)
    start, end = (page - 1) * size, (page - 1) * size + size
    view = rows[start:end]

    previews = []
    for ridx, r in enumerate(view, start=start):
        if "__error__" in r:
            previews.append({"idx": ridx, "preview": f'<pre class="err">{_html.escape(r["__error__"])}\n{_html.escape(r.get("__raw__",""))}</pre>'})
            continue
        if "messages" in r and isinstance(r["messages"], list):
            html = render_messages_html(r["messages"])
            previews.append({"idx": ridx, "preview": html})
        else:
            txt = json.dumps(r, ensure_ascii=False, indent=2)
            previews.append({"idx": ridx, "preview": f"<pre>{highlight_pii(_html.escape(txt))}</pre>"})

    return templates.TemplateResponse("file.html", {
        "request": request,
        "name": name,
        "total": total,
        "page": page,
        "size": size,
        "q": q,
        "regex": regex,
        "previews": previews
    })

@app.get("/api/get_record")
def api_get_record(name: str, idx: int):
    path = DATA_DIR / name
    rows = read_jsonl(path)
    if idx < 0 or idx >= len(rows):
        raise HTTPException(404, "Index out of range")
    return rows[idx]

@app.post("/api/update_record")
def api_update_record(name: str = Form(...), idx: int = Form(...), payload: str = Form(...)):
    path = DATA_DIR / name
    rows = read_jsonl(path)
    if idx < 0 or idx >= len(rows):
        raise HTTPException(404, "Index out of range")
    try:
        obj = json.loads(payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"JSON parse error: {e}"}, status_code=400)
    rows[idx] = obj
    write_jsonl(path, rows)
    return {"ok": True, "idx": idx}

@app.post("/api/add_record")
def api_add_record(name: str = Form(...), payload: str = Form(...)):
    path = DATA_DIR / name
    rows = read_jsonl(path)
    try:
        obj = json.loads(payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"JSON parse error: {e}"}, status_code=400)
    rows.append(obj)
    write_jsonl(path, rows)
    return {"ok": True, "idx": len(rows) - 1}

@app.post("/api/delete_record")
def api_delete_record(name: str = Form(...), idx: int = Form(...)):
    path = DATA_DIR / name
    rows = read_jsonl(path)
    if idx < 0 or idx >= len(rows):
        raise HTTPException(404, "Index out of range")
    del rows[idx]
    write_jsonl(path, rows)
    return {"ok": True}

# --------------------------------------------------------------------------------------
# Templates (written to disk on first run)
# --------------------------------------------------------------------------------------
TEMPLATES = BASE / "templates"
TEMPLATES.mkdir(parents=True, exist_ok=True)

(TEMPLATES / "index.html").write_text("""<!doctype html>
<html><head>
  <meta charset="utf-8"/><title>{{request.app.title}}</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  <link rel="stylesheet" href="/static/style.css">
</head><body>
  <header><h1>JSONL Editor</h1><span class="muted">Data dir: {{data_dir}}</span></header>
  <aside>
    <h3>Files</h3>
    <ul>
      {% for f in files %}
      <li><a href="/file/{{f}}">{{f}}</a></li>
      {% else %}
      <li class="muted">No *.jsonl files found</li>
      {% endfor %}
    </ul>
  </aside>
  <main>
    <p>Select a file to view records.</p>
  </main>
</body></html>
""", encoding="utf-8")

(TEMPLATES / "file.html").write_text("""<!doctype html>
<html><head>
  <meta charset="utf-8"/><title>{{name}} — JSONL</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  <link rel="stylesheet" href="/static/style.css">
</head><body>
  <header class="row">
    <h1>{{name}}</h1>
    <nav><a href="/">← files</a></nav>
    <form method="get" action="/file/{{name}}" class="search">
      <input type="hidden" name="size" value="{{size}}">
      <input name="q" value="{{q}}" placeholder="search text or regex">
      <label title="use Python-style regex"><input type="checkbox" name="regex" value="true" {% if regex %}checked{% endif %}> regex</label>
      <button>Search</button>
    </form>
  </header>

  <section class="pager">
    <span>Total: {{total}}</span>
    <span>Page:
      {% if page>1 %}
      <a href="/file/{{name}}?page={{page-1}}&size={{size}}&q={{q}}&regex={{regex}}">« Prev</a>
      {% endif %}
      <b>{{page}}</b>
      {% if page*size < total %}
      <a href="/file/{{name}}?page={{page+1}}&size={{size}}&q={{q}}&regex={{regex}}">Next »</a>
      {% endif %}
    </span>
    <span>Size:
      <a href="/file/{{name}}?page=1&size=10&q={{q}}&regex={{regex}}">10</a> |
      <a href="/file/{{name}}?page=1&size=20&q={{q}}&regex={{regex}}">20</a> |
      <a href="/file/{{name}}?page=1&size=50&q={{q}}&regex={{regex}}">50</a>
    </span>
  </section>

  <section class="records">
    {% for item in previews %}
    <article class="card">
      <header>#{{item.idx}}
        <button hx-get="/api/get_record?name={{name}}&idx={{item.idx}}"
                hx-target="#editor"
                hx-swap="innerHTML">Edit</button>
        <form hx-post="/api/delete_record" hx-vals='{"name":"{{name}}","idx":"{{item.idx}}"}'
              hx-confirm="Delete this record?" hx-target="closest article" hx-swap="outerHTML">
          <button class="danger">Delete</button>
        </form>
      </header>
      {{item.preview|safe}}
    </article>
    {% endfor %}
  </section>

  <section id="editor" class="editor">
    <h3>Add / Edit</h3>
    <form hx-post="/api/add_record" hx-target="#status" hx-swap="innerHTML">
      <input type="hidden" name="name" value="{{name}}">
      <textarea name="payload" rows="12" spellcheck="false">{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Prompt here"},{"role":"assistant","content":"Answer here"}]}</textarea>
      <button>Add Record</button>
    </form>
    <div id="status"></div>
  </section>
</body></html>
""", encoding="utf-8")

# --------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------
(static_dir / "style.css").write_text("""
:root { --bg:#0b0c10; --fg:#e5e7eb; --muted:#9aa0a6; --acc:#6ee7b7; --warn:#f59e0b; --err:#ef4444; }
* { box-sizing: border-box; }
body { margin:0; background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
header { padding: 12px 16px; display:flex; gap:16px; align-items:center; border-bottom:1px solid #111827; }
header.row { justify-content: space-between; }
a { color: var(--acc); text-decoration: none; }
.muted { color: var(--muted); font-size: 12px; }
aside { float:left; width: 220px; border-right:1px solid #111827; height: 100vh; overflow:auto; padding: 12px; }
aside ul { list-style: none; padding: 0; margin: 0; }
aside li { margin: 6px 0; }
main, section { padding: 12px 16px; }
.search input[name=q] { width: 320px; }
.pager { display:flex; gap: 24px; align-items:center; border-bottom:1px solid #111827; padding: 10px 16px; }
.records { display:grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap: 12px; padding: 12px; }
.card { background:#111827; border:1px solid #1f2937; border-radius:12px; padding: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.25); }
.card header { display:flex; justify-content: space-between; align-items:center; border:0; padding: 0 0 6px 0; }
pre { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; background:#0f172a; padding:8px; border-radius:8px; }
.err { color: var(--err); }
button { background:#1f2937; color: #e5e7eb; border:1px solid #374151; padding:6px 10px; border-radius:8px; cursor:pointer; }
button:hover { background:#374151; }
button.danger { border-color:#7f1d1d; }
.editor textarea { width: 100%; background:#0f172a; color:#e5e7eb; border:1px solid #1f2937; border-radius:8px; padding:8px; }

mark.pii { background: #3f6212; color:#fff; padding:0 2px; border-radius:3px; }

/* Role colors */
.role-system   { --b:#7c3aed; --bg:#1f1536; }  /* purple  */
.role-user     { --b:#2563eb; --bg:#0f1a33; }  /* blue    */
.role-assistant{ --b:#10b981; --bg:#0d1f1a; }  /* green   */
.role-unknown  { --b:#9ca3af; --bg:#111827; }  /* gray    */

/* Bubble layout */
.bubble {
  border: 1px solid var(--b);
  background: var(--bg);
  border-radius: 12px;
  margin: 8px 0;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.25);
}
.bubble-head {
  font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--b); padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,.06);
  background: rgba(0,0,0,.15);
}
.bubble-body { padding: 10px 12px; white-space: pre-wrap; word-wrap: break-word; }
""", encoding="utf-8")

# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[jsonl_ui] serving {DATA_DIR} on http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

