import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.api.routes_zones import router as zones_router
from backend.api.routes_incidents import router as incidents_router
from backend.api.routes_stream import router as stream_router
from backend.api.routes_learning import router as learning_router
from backend.api.routes_rules import router as rules_router
from backend.api.routes_system import router as system_router
from backend.api.routes_executive import router as executive_router

app = FastAPI(
    title="РВК Safety | Сервер видеоаналитики ТБ",
    description="Комплексная система мониторинга СИЗ и опасных зон для ООО «РВК.Цифровые решения» (ГК «Росводоканал»)",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(zones_router)
app.include_router(incidents_router)
app.include_router(stream_router)
app.include_router(learning_router)
app.include_router(rules_router)
app.include_router(system_router)
app.include_router(executive_router)

# Mount frontend
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount active learning store media
STORE_DIR = os.path.join(BASE_DIR, "backend", "data", "active_learning_store")
os.makedirs(STORE_DIR, exist_ok=True)
app.mount("/learning_store", StaticFiles(directory=STORE_DIR), name="learning_store")

# Mount training reports & validation curves
REPORTS_DIR = os.path.join(BASE_DIR, "dataset", "previews", "training_reports")
if os.path.exists(REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Mount dataset crops & slicing viewer
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CROPS_DIR = os.path.join(DATASET_DIR, "crops")
if os.path.exists(CROPS_DIR):
    app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")

HEAD_CROPS_DIR = os.path.join(DATASET_DIR, "head_crops")
if os.path.exists(HEAD_CROPS_DIR):
    app.mount("/head_crops", StaticFiles(directory=HEAD_CROPS_DIR), name="head_crops")

PREVIEWS_DIR = os.path.join(BASE_DIR, "dataset", "previews")
if os.path.exists(PREVIEWS_DIR):
    app.mount("/previews", StaticFiles(directory=PREVIEWS_DIR), name="previews")

@app.get("/dataset")
def read_dataset_viewer():
    viewer_path = os.path.join(DATASET_DIR, "crops_viewer.html")
    if os.path.exists(viewer_path):
        resp = FileResponse(viewer_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return {"status": "error", "message": "Dataset viewer not found"}

@app.get("/tagger")
def read_tagger():
    tagger_path = os.path.join(DATASET_DIR, "fast_tagger.html")
    if os.path.exists(tagger_path):
        resp = FileResponse(tagger_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return {"status": "error", "message": "Fast tagger not found"}

@app.post("/save_labels")
async def save_labels(payload: dict):
    """Save verified labels directly from Fast Tagger interface."""
    import json
    out_path = os.path.join(DATASET_DIR, "verified_labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"status": "ok", "saved_items": len(payload.get("items", []))}

@app.get("/api/dataset/verified")
def get_verified_dataset():
    """Retrieve verified crop labels and statuses from dataset/verified_labels.json."""
    verified_path = os.path.join(DATASET_DIR, "verified_labels.json")
    if os.path.exists(verified_path):
        import json
        with open(verified_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total": 0, "items": []}


@app.get("/")
def read_root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        resp = FileResponse(index_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
@app.get("/player")
def read_player():
    player_path = os.path.join(DATASET_DIR, "previews", "video_player.html")
    if os.path.exists(player_path):
        resp = FileResponse(player_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return {"status": "error", "message": "Player not found"}

@app.get("/presentation")
def read_presentation():
    pres_path = os.path.join(FRONTEND_DIR, "presentation.html")
    if os.path.exists(pres_path):
        resp = FileResponse(pres_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return {"status": "error", "message": "Presentation not found"}

@app.get("/report")
def read_report():
    dash_path = os.path.join(REPORTS_DIR, "dashboard.html")
    if os.path.exists(dash_path):
        resp = FileResponse(dash_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return {"status": "error", "message": "Report not found"}

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    import socket

    def find_port(preferred=8000, fallbacks=(8088, 8081)):
        for p in [preferred, *fallbacks]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        return preferred

    port = int(os.environ.get("PORT", find_port()))
    print(f"\n=======================================================")
    print(f"🚀 RVK Safety CV server starting on http://localhost:{port}")
    print(f"=======================================================\n")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
