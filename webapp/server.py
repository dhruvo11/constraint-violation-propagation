"""The CVP web app: a FastAPI server for the page in webapp/static and its JSON API.

  .venv/bin/python -m webapp            http://127.0.0.1:7860 (PORT and HOST change it)

Everything runs on this machine; nothing is sent anywhere. The models load in the background
when the server starts, then the guided tour is computed so that it opens instantly.
"""
import io
import json
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from cvp import doctypes, paths

from . import samples
from .engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
ENGINE = Engine()
MAX_BYTES = 20 * 1024 * 1024
CATEGORY_LABELS = {"genuine": "Genuine (no edit)", "T1": "Obvious edit (pasted text)", "T2": "Smooth AI-style edit",
                   "T3": "Obvious edit, totals fixed too", "T4": "Smooth edit, totals fixed too"}


def _precompute():
    """Run the guided tour, then the other examples, so that clicking them is instant."""
    av = ENGINE.availability()
    order = [s for g in samples.catalogue(av) for s in g["items"] if s["available"]]
    for s in order:
        if s["id"] in ENGINE.cache:
            continue
        with ENGINE.lock:
            try:
                ENGINE.cache[s["id"]] = samples.run(ENGINE, samples.find(s["id"]))
            except Exception as e:  # a broken example must not stop the others
                print(f"example {s['id']} failed: {type(e).__name__}: {e}", flush=True)


@asynccontextmanager
async def lifespan(app):
    ENGINE.warm_up(then=_precompute)
    yield


app = FastAPI(title="CVP", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=os.path.join(paths.REPORT, "final_report", "figures")), name="figures")


@app.get("/")
def index():
    """The page, with its script and stylesheet stamped by version so a browser never keeps an old one."""
    html = open(os.path.join(STATIC, "index.html")).read()
    for name in ("app.js", "app.css"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int(os.path.getmtime(os.path.join(STATIC, name)))}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/explainer")
def explainer():
    return FileResponse(os.path.join(paths.BASE, "docs", "how-it-works.html"))


@app.get("/api/status")
def status():
    return dict(ENGINE.state, available=ENGINE.availability(), cached=len(ENGINE.cache))


@app.get("/api/catalog")
def catalog():
    av = ENGINE.availability()
    types = []
    for n in doctypes.names():
        DT = doctypes.get(n)
        S = DT.schema
        types.append(dict(name=n, title=DT.title, available=av["synthetic"].get(n, False),
                          fields=[dict(name=f.name, label=f.label) for f in S.fields if S.constrained(f.name)]))
    return dict(groups=samples.catalogue(av), doctypes=types, categories=CATEGORY_LABELS,
                cord=dict(available=av["cord"], receipts=av["cord_receipts"]), receipt=av["receipt"])


@app.get("/api/cord_fields/{k}")
def cord_fields(k: int):
    from cvp.real import cord, forge
    from cvp.real import receipt_schema as RS
    if not ENGINE.availability()["cord"]:
        raise HTTPException(404, "CORD is not available on this machine.")
    r = cord.receipt("test", max(0, min(cord.count_receipts("test") - 1, k)))
    labels = {f.name: f.label for f in RS.schema(cord.from_lines(r.lines)).fields}
    return [dict(name=f, label=labels[f]) for f in forge.editable(r)]


def _run(fn, *a, **k):
    with ENGINE.lock:
        try:
            return fn(*a, **k)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"The analysis failed: {type(e).__name__}: {e}")


@app.post("/api/sample/{sid}")
def run_sample(sid: str):
    if sid in ENGINE.cache:
        return ENGINE.cache[sid]
    try:
        s = samples.find(sid)
    except KeyError:
        raise HTTPException(404, "No such example.")
    out = _run(samples.run, ENGINE, s)
    ENGINE.cache[sid] = out
    return out


@app.post("/api/forge")
def forge(doctype: str = Body(...), seed: int = Body(0), category: str = Body("T2"), field: str = Body(None)):
    if category not in CATEGORY_LABELS:
        raise HTTPException(400, "Unknown kind of forgery.")
    field = field or None
    if doctype == "cord":
        return _run(ENGINE.forge_cord, seed, category, field)
    if doctype not in doctypes.names():
        raise HTTPException(400, "Unknown document type.")
    return _run(ENGINE.forge_synthetic, doctype, seed, category, field)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), kind: str = Form("receipt")):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That file is larger than 20 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")  # phone photos carry their rotation in EXIF
    except Exception:
        raise HTTPException(400, "That file is not an image this app can open. Use a JPEG or PNG "
                                 "(iPhone HEIC photos: export them as JPEG first).")
    arr = np.asarray(im)
    name = os.path.basename(file.filename or "upload")
    if kind == "receipt":
        return await _in_thread(lambda: _run(ENGINE.analyse_receipt, arr, title=name))
    if kind not in doctypes.names():
        raise HTTPException(400, "Unknown document type.")
    return await _in_thread(lambda: _run(ENGINE.analyse_synthetic, kind, arr))


async def _in_thread(fn):
    """Run blocking work off the event loop."""
    import anyio
    return await anyio.to_thread.run_sync(fn)


@app.post("/api/recheck")
def recheck(id: str = Body(...), values: dict = Body(...)):
    if id not in ENGINE._kept:
        raise HTTPException(404, "This reading has expired; analyse the photo again.")
    for f, v in values.items():
        try:
            if v not in (None, ""):
                float(str(v).replace(",", ""))
        except ValueError:
            raise HTTPException(400, f"“{v}” is not a number.")
    return _run(ENGINE.recheck, id, values)


@app.get("/api/results")
def results():
    """Headline numbers of the evaluation, read from the saved results."""
    out = {}
    R = os.path.join(paths.BASE, "results")
    for n in doctypes.names():
        p = os.path.join(R, n, "results.json")
        if os.path.exists(p):
            res = json.load(open(p))["results"]
            out.setdefault("synthetic", {})[doctypes.get(n).title] = {
                m: dict(AUC=res[m]["AUC"], attr=res[m]["attr_all"]) for m in ("Visual-only", "Constraint-only", "CVP")
                if m in res}
    for tag, label in (("", "zero-shot"), ("_crude", "fine-tuned (crude edits)"), ("_all", "fine-tuned (all)")):
        p = os.path.join(R, "real", f"cord_cvp_test{tag}.json")
        if os.path.exists(p):
            res = json.load(open(p))["results"]
            out.setdefault("cord", {})[label] = {m: dict(AUC=res[m]["AUC"], attr=res[m]["attr_all"])
                                                 for m in ("Visual-only", "Constraint-only", "CVP")}
    for v, label in (("v2", "GPT-Image-2"), ("v1", "Gemini / Ideogram")):
        p = os.path.join(R, "real", f"aiforge_{v}.json")
        if os.path.exists(p):
            res = json.load(open(p))["results"]["zero-shot"]["methods"]
            out.setdefault("aiforge", {})[label] = {m: dict(AUC=x["AUC"], attr=x["attr"]) for m, x in res.items()}
    p = os.path.join(R, "real", "upload_check.json")
    if os.path.exists(p):
        out["experimental"] = json.load(open(p))["summary"]
    return JSONResponse(out)


def main():
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 7860))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    if not os.environ.get("CVP_NO_BROWSER"):
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"CVP is starting at {url} (the models load in the background; the page shows when they are ready)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
