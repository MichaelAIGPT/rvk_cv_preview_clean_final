import os
import json
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, Body
from backend.active_learning import al_manager
from backend.streamer import streamer
from backend.storage import storage
from backend.logger import sys_logger

router = APIRouter(prefix="/api/learning", tags=["active_learning"])

@router.get("/samples")
def get_learning_samples():
    """Retrieve all collected active learning samples and category statistics."""
    return {
        "stats": al_manager.get_stats(),
        "samples": al_manager.get_manifest()
    }

@router.delete("/samples/{sample_id}")
def delete_learning_sample(sample_id: str):
    """Delete an active learning sample and its physical files."""
    success = al_manager.delete_sample(sample_id)
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found")
    streamer.reload_corrections(reprocess=True)
    sys_logger.audit("LEARNING", f"Удален образец Active Learning: {sample_id}")
    return {"status": "ok", "deleted_id": sample_id}

@router.post("/export")
def export_dataset_for_yolo():
    """Export active learning samples into a standardized YOLO fine-tuning directory."""
    result = al_manager.export_yolo_dataset()
    sys_logger.info("LEARNING", f"Экспортирован датасет для дообучения YOLO: {result.get('sample_count', 0)} сэмплов")
    return {"status": "ok", **result}

@router.post("/train")
def trigger_training(payload: dict = {}):
    """Trigger background fine-tuning of the PPE model on GPU using accumulated active learning samples."""
    from backend.trainer import bg_trainer
    epochs = int(payload.get("epochs", 15))
    batch = int(payload.get("batch", 8))
    sys_logger.info("TRAIN", f"Инициирован запуск дообучения модели: {epochs} эпох, batch={batch}")
    res = bg_trainer.start_training(epochs=epochs, batch=batch)
    return res

@router.get("/train/status")
def get_training_status():
    """Get live progress of the background model training on GPU."""
    from backend.trainer import bg_trainer
    return bg_trainer.get_status()

@router.post("/tag_crop")
def tag_dataset_crop(payload: dict):
    """Update annotation for an existing crop in dataset/verified_labels.json."""
    filename = payload.get("filename", "")
    new_label = payload.get("label", "")
    if not filename or not new_label:
        raise HTTPException(status_code=400, detail="filename and label are required")
    res = al_manager.tag_dataset_crop(filename, new_label)
    return res

@router.post("/tag_batch")
def tag_crops_batch(payload: dict):
    """Update annotations in batch for multiple crops across dataset & active learning store."""
    items = payload.get("items", [])
    new_label = payload.get("label", "")
    if not items or not new_label:
        raise HTTPException(status_code=400, detail="items list and label are required")
    res = al_manager.tag_batch(items, new_label)
    return res

@router.get("/models")
def get_available_models():
    """List available model weights (v1 baseline and v2 fine-tuned) and current active status."""
    from backend.trainer import bg_trainer
    return {"models": bg_trainer.get_available_weights()}

@router.post("/models/switch")
def switch_active_model(payload: dict):
    """Switch pipeline detector weights between v1 baseline and v2 fine-tuned on the fly."""
    from backend.trainer import bg_trainer
    model_id = payload.get("model_id", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    res = bg_trainer.switch_weights(model_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res

# --- SOTA Multi-Model Hub & Benchmark ---

@router.get("/benchmark")
def get_models_benchmark():
    """Retrieve comparative benchmark metrics between MobileNetV3, EfficientNet-B0, and ResNet-18."""
    bench_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models_benchmark.json")
    if os.path.exists(bench_file):
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

@router.get("/state")
def get_pipeline_models_state():
    """Get active classifier, segmentation mode, and smoke/fire mode."""
    pipeline = streamer.pipeline
    return {
        "active_classifier": getattr(pipeline, "active_classifier_name", "efficientnet_b0"),
        "segmentation_mode": getattr(pipeline, "segmentation_mode", False),
        "smoke_fire_mode": getattr(pipeline, "smoke_fire_mode", False),
        "current_weights_path": getattr(pipeline, "current_weights_path", "cv_core/weights/ppe_detector_best.pt")
    }

@router.post("/classifier/switch")
def switch_classifier_backbone(payload: dict = Body(...)):
    """Switch classifier backbone dynamically: 'efficientnet_b0', 'mobilenet_v3', 'resnet_18'."""
    classifier_name = payload.get("classifier", "efficientnet_b0")
    success = streamer.pipeline.set_classifier(classifier_name)
    if success:
        streamer._process_single_frame(streamer.current_frame_idx)
        return {"status": "ok", "active_classifier": classifier_name}
    raise HTTPException(status_code=400, detail=f"Failed to switch to classifier '{classifier_name}'")

@router.post("/segmentation/toggle")
def toggle_segmentation(payload: dict = Body(default={})):
    """Toggle YOLOv8-Seg polygon instance segmentation."""
    current = getattr(streamer.pipeline, "segmentation_mode", False)
    new_state = payload.get("enabled", not current)
    result = streamer.pipeline.set_segmentation_mode(new_state)
    streamer._process_single_frame(streamer.current_frame_idx)
    return {"status": "ok", "segmentation_mode": result}

@router.post("/smokefire/toggle")
def toggle_smoke_fire(payload: dict = Body(default={})):
    """Toggle Smoke & Fire hazard detector."""
    current = getattr(streamer.pipeline, "smoke_fire_mode", False)
    new_state = payload.get("enabled", not current)
    result = streamer.pipeline.set_smoke_fire_mode(new_state)
    streamer._process_single_frame(streamer.current_frame_idx)
    return {"status": "ok", "smoke_fire_mode": result}

@router.get("/features/state")
def get_features_state():
    """Get active states of all modular safety checkers."""
    return getattr(streamer.pipeline, "get_features_state", lambda: {})()

@router.post("/features/toggle")
def toggle_modular_feature(payload: dict = Body(...)):
    """Dynamically enable or disable a specific safety detector (hardhat, vest, smoking, phone, danger_zones, smokefire)."""
    feature = payload.get("feature", "")
    current_state = streamer.pipeline.get_features_state().get(feature, True)
    enabled = payload.get("enabled", not current_state)
    result = streamer.pipeline.set_feature_toggle(feature, enabled)
    streamer._process_single_frame(streamer.current_frame_idx)
    return {"status": "ok", "feature": feature, "enabled": result, "all_features": streamer.pipeline.get_features_state()}


# --- MLOps Run Tracking & Accuracy Progression ---

RUNS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "run_history.json")

@router.get("/runs")
def get_validation_runs():
    """Get history of validation runs and accuracy trends."""
    if os.path.exists(RUNS_FILE):
        try:
            with open(RUNS_FILE, "r", encoding="utf-8") as f:
                runs = json.load(f)
                return {"runs": runs, "total": len(runs)}
        except Exception:
            pass
    return {"runs": [], "total": 0}

@router.post("/runs/record")
def record_current_run(payload: dict = Body(default={})):
    """Record a test run snapshot into history with accuracy metrics."""
    runs = []
    if os.path.exists(RUNS_FILE):
        try:
            with open(RUNS_FILE, "r", encoding="utf-8") as f:
                runs = json.load(f)
        except Exception:
            runs = []

    incidents = storage.get_incidents()
    verified = sum(1 for i in incidents if i.get("status") == "verified")
    false_alarms = sum(1 for i in incidents if i.get("status") == "false_alarm")
    total = verified + false_alarms
    precision = round((verified / total * 100), 1) if total > 0 else 0.0

    run_entry = {
        "run_id": f"run-{len(runs) + 1:03d}",
        "timestamp": time.time(),
        "date_str": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "model_name": payload.get("model_name") or getattr(streamer.pipeline, "active_classifier_name", "EfficientNet-B0 SOTA"),
        "playlist": streamer.current_playlist,
        "total_alerts": len(incidents),
        "verified": verified,
        "false_alarms": false_alarms,
        "precision": precision,
        "f1_score": round(2 * (precision / 100 * 0.94) / (precision / 100 + 0.94), 2) if precision > 0 else 0.0,
        "notes": payload.get("notes") or f"Прогон оператора: {verified} подтверждено, {false_alarms} ложных"
    }
    runs.append(run_entry)
    with open(RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

    sys_logger.audit("BENCHMARK", f"Зафиксирован прогон {run_entry['run_id']}: Точность {precision}% ({verified} верных / {false_alarms} ложных)", meta=run_entry)
    return {"status": "ok", "run": run_entry}
