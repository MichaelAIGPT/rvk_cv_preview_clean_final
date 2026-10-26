import os
import yaml
from fastapi import APIRouter, Body
from backend.streamer import streamer

router = APIRouter(prefix="/api/rules", tags=["Safety Rules"])

RULES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "safety_rules.yaml")

DEFAULT_RULES = {
    "version": "1.2.0",
    "facility": "Цех механического обезвоживания осадка (ЦМОО)",
    "rules": {
        "danger_zone": {
            "name": "Нахождение в опасной зоне механизмов",
            "time_threshold_seconds": 5.0,
            "severity": "CRITICAL",
            "cooldown_seconds": 60,
            "sound_alarm": True,
            "description": "Фиксация нахождения персонала в зоне шнеков/сепараторов дольше допустимого"
        },
        "no_hardhat": {
            "name": "Отсутствие защитной каски",
            "time_threshold_seconds": 3.0,
            "severity": "WARNING",
            "cooldown_seconds": 60,
            "sound_alarm": False,
            "description": "Сглаженное отсутствие СИЗ головы при устойчивом треке человека"
        },
        "smoking": {
            "name": "Контроль курения в неположенном месте",
            "time_threshold_seconds": 2.5,
            "severity": "CRITICAL",
            "cooldown_seconds": 120,
            "sound_alarm": True,
            "description": "Детекция жеста рука-рот в производственной зоне"
        },
        "safety_vest": {
            "name": "Отсутствие сигнального жилета",
            "time_threshold_seconds": 4.0,
            "severity": "WARNING",
            "cooldown_seconds": 60,
            "sound_alarm": False,
            "description": "Контроль ношения светоотражающего жилета в транспортных проездах"
        }
    },
    "inference": {
        "confidence_threshold": 0.40,
        "iou_threshold": 0.45,
        "person_min_height_px": 60,
        "model_tag": "construction-ppe-960"
    }
}

def load_rules():
    if not os.path.exists(RULES_FILE):
        os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_RULES, f, allow_unicode=True, default_flow_style=False)
        return DEFAULT_RULES
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or DEFAULT_RULES
    except Exception as e:
        print("Error reading safety_rules.yaml:", e)
        return DEFAULT_RULES

def save_rules(rules_data):
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        yaml.dump(rules_data, f, allow_unicode=True, default_flow_style=False)

@router.get("")
def get_safety_rules():
    """Get current YAML event and safety rules configuration per TZ 4.1.7 & 4.2.4."""
    return load_rules()

@router.post("")
def update_safety_rules(payload: dict = Body(...)):
    """Update YAML event rules and apply thresholds to pipeline."""
    current = load_rules()
    if "rules" in payload:
        current["rules"].update(payload["rules"])
    if "inference" in payload:
        current["inference"].update(payload["inference"])
    save_rules(current)

    # Sync into pipeline if applicable
    dz = current.get("rules", {}).get("danger_zone", {})
    if "time_threshold_seconds" in dz:
        setattr(streamer.pipeline, "danger_zone_time_threshold", float(dz["time_threshold_seconds"]))
    nh = current.get("rules", {}).get("no_hardhat", {})
    if "time_threshold_seconds" in nh:
        setattr(streamer.pipeline, "no_hardhat_time_threshold", float(nh["time_threshold_seconds"]))

    return {"status": "ok", "message": "Правила успешно обновлены", "rules": current}
