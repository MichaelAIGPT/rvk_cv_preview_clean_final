from fastapi import APIRouter, HTTPException
from typing import List
from backend.models import ZoneCreate, ZoneOut
from backend.storage import storage

router = APIRouter(prefix="/api/zones", tags=["Zones"])

@router.get("", response_model=List[ZoneOut])
def get_zones():
    return storage.get_zones()

@router.post("", response_model=ZoneOut)
def create_zone(zone: ZoneCreate):
    created = storage.add_zone(zone.model_dump())
    return created

@router.post("/{zone_id}/toggle", response_model=ZoneOut)
def toggle_zone(zone_id: str):
    updated = storage.toggle_zone(zone_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Zone not found")
    return updated

@router.delete("/{zone_id}")
def delete_zone(zone_id: str):
    success = storage.delete_zone(zone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"status": "success", "deleted_id": zone_id}
