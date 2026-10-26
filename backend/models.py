from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Point(BaseModel):
    x: float
    y: float

class ZoneCreate(BaseModel):
    name: str = Field(..., description="Название зоны, например: Центробежный насос №1")
    polygon: List[List[float]] = Field(..., description="Координаты вершин полигона [[x, y], ...]")
    is_active: bool = Field(True, description="Активность зоны")
    color: Optional[str] = Field("#ef4444", description="Цвет подсветки зоны в HEX")

class ZoneOut(ZoneCreate):
    id: str = Field(..., description="Уникальный ID зоны")

class IncidentVerify(BaseModel):
    status: str = Field(..., description="'verified', 'false_alarm' или 'new'")
    notes: Optional[str] = Field(None, description="Комментарий оператора")
    violations: Optional[List[str]] = Field(None, description="Скорректированный список нарушений")

class CanonicalHSEInvestigation(BaseModel):
    why1: Optional[str] = Field(None, description="Почему 1: Первичная причина")
    why2: Optional[str] = Field(None, description="Почему 2: Поведение/Условие")
    why3: Optional[str] = Field(None, description="Почему 3: Недостаток контроля/СИЗ")
    why4: Optional[str] = Field(None, description="Почему 4: Организационный пробел")
    why5: Optional[str] = Field(None, description="Почему 5: Системная коренная причина")
    root_cause: Optional[str] = Field(None, description="Итоговая коренная причина инцидента")
    corrective_action: Optional[str] = Field(None, description="Корректирующее и предупреждающее действие (CAPA)")
    responsible_person: Optional[str] = Field(None, description="Ответственное лицо (ФИО, должность)")
    deadline: Optional[str] = Field(None, description="Срок устранения / выполнения мер")

class IncidentHSEUpdate(BaseModel):
    status: str = Field(..., description="Статус: 'triage', 'investigation', 'capa', 'closed', 'false_alarm'")
    notes: Optional[str] = None
    role: Optional[str] = "hse_specialist"
    violations: Optional[List[str]] = None
    investigation: Optional[CanonicalHSEInvestigation] = None

class IncidentOut(BaseModel):
    id: str
    timestamp: float
    time_str: str
    frame_idx: int
    track_id: Any
    violations: List[str]
    bbox: List[int]
    status: str = "new"  # 'new', 'triage', 'investigation', 'capa', 'closed', 'false_alarm', 'verified'
    notes: Optional[str] = None
    playlist: Optional[str] = "golden"
    camera_title: Optional[str] = "ЦМОО • Сепараторы"
    model_version: Optional[str] = "v3_cascade"
    investigation: Optional[CanonicalHSEInvestigation] = None
    history: Optional[List[Dict[str, Any]]] = None
