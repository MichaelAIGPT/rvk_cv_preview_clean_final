import os
import cv2
import csv
import io
import time
from fastapi import APIRouter, HTTPException, Query, Response
from typing import List, Optional
from backend.models import IncidentOut, IncidentVerify, IncidentHSEUpdate
from backend.storage import storage
from backend.logger import sys_logger

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])

@router.get("", response_model=List[IncidentOut])
def get_incidents(status: Optional[str] = Query(None, description="Фильтр статуса")):
    all_inc = storage.get_incidents()
    if status:
        return [i for i in all_inc if i.get("status") == status]
    return all_inc

@router.post("/{incident_id}/verify", response_model=IncidentOut)
def verify_incident(incident_id: str, payload: IncidentVerify):
    updated = storage.verify_incident(
        incident_id,
        status=payload.status,
        notes=payload.notes,
        role="operator",
        violations=payload.violations
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")

    status_ru = "Подтверждено" if payload.status == "verified" else ("Ложное срабатывание" if payload.status == "false_alarm" else payload.status)
    sys_logger.audit("TRIAGE", f"Инцидент {incident_id} проверен оператором: {status_ru}", meta={"incident_id": incident_id, "status": payload.status, "notes": payload.notes})

    # If marked as false_alarm, auto-mine crop as hard negative for Active Learning!
    if payload.status == "false_alarm" and updated.get("bbox"):
        try:
            from backend.streamer import streamer
            from backend.active_learning import al_manager
            frame_idx = updated.get("frame_idx", 0)
            if streamer.frame_paths and 0 <= frame_idx < len(streamer.frame_paths):
                img_path = streamer.frame_paths[frame_idx]
                frame_bgr = cv2.imread(img_path)
                if frame_bgr is not None:
                    al_manager.save_sample(
                        frame_bgr=frame_bgr,
                        bbox=updated["bbox"],
                        user_label="not_person_pipe",
                        source_frame_name=os.path.basename(img_path),
                        playlist=streamer.current_playlist,
                        track_id=str(updated.get("track_id", "false_alarm")),
                        model_prediction={"has_hardhat": False, "confidence": 0.5}
                    )
        except Exception as e:
            print(f"Error auto-mining hard negative for incident {incident_id}: {e}")

    return updated

@router.post("/{incident_id}/hse", response_model=IncidentOut)
def update_incident_hse_workflow(incident_id: str, payload: IncidentHSEUpdate):
    """Canonical HSE Incident Lifecycle: TRIAGE -> INVESTIGATION (5 Whys) -> CAPA -> CLOSED."""
    inv_dict = payload.investigation.dict() if payload.investigation else None
    updated = storage.update_incident_hse(
        inc_id=incident_id,
        status=payload.status,
        notes=payload.notes,
        role=payload.role or "hse_specialist",
        investigation=inv_dict,
        violations=payload.violations
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")

    sys_logger.audit("CAPA", f"Инцидент {incident_id}: обновлено расследование/меры (статус: {payload.status})", meta={"incident_id": incident_id, "status": payload.status, "role": payload.role})
    return updated

@router.get("/export/scada")
def export_scada_telemetry():
    """Real-time SCADA / OPC UA compatibility endpoint."""
    from backend.streamer import streamer
    state = streamer.get_frame_state()
    incidents = storage.get_incidents()
    
    active_no_hh = sum(1 for d in state.get("detections", []) if not d.get("has_hardhat", True))
    danger_zone_active = any(d.get("in_danger_zone", False) for d in state.get("detections", []))
    fire_alert = any("FIRE_DETECTED" in inc.get("violations", []) for inc in incidents[:5])
    smoke_alert = any("SMOKE_DETECTED" in inc.get("violations", []) for inc in incidents[:5])
    
    alert_level = "NORMAL"
    if fire_alert:
        alert_level = "CRITICAL_FIRE"
    elif smoke_alert or (active_no_hh > 0 and danger_zone_active):
        alert_level = "HIGH_HAZARD"
    elif active_no_hh > 0 or danger_zone_active:
        alert_level = "WARNING"

    return {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subsystem": "ROSVODOKANAL_CV_SAFETY_v2",
        "camera_id": "CAM-01-PUMP-HALL",
        "camera_location": "Цех механического обезвоживания осадка (ЦМОО)",
        "telemetry": {
            "workers_detected": len(state.get("detections", [])),
            "no_hardhat_count": active_no_hh,
            "danger_zone_breach": danger_zone_active,
            "fire_detected": fire_alert,
            "smoke_detected": smoke_alert,
            "alert_level": alert_level,
            "pipeline_fps": streamer.fps,
            "pipeline_latency_ms": streamer.latency_ms,
            "active_classifier": getattr(streamer.pipeline, "active_classifier_name", "efficientnet_b0"),
            "segmentation_active": getattr(streamer.pipeline, "segmentation_mode", False),
            "smoke_fire_active": getattr(streamer.pipeline, "smoke_fire_mode", False)
        },
        "opc_ua_tags": {
            "DB100.SAFETY_ALERT_INT": 3 if alert_level.startswith("CRITICAL") else (2 if alert_level == "HIGH_HAZARD" else (1 if alert_level == "WARNING" else 0)),
            "DB100.WORKER_COUNT": len(state.get("detections", [])),
            "DB100.PPE_VIOLATION_BOOL": active_no_hh > 0,
            "DB100.ZONE_BREACH_BOOL": danger_zone_active
        }
    }

@router.get("/export/director")
def export_director_report(
    format: str = Query("xlsx", description="'xlsx', 'csv' или 'json'"),
    period: str = Query("all", description="'shift', '7d', '30d', 'all'")
):
    """Director / 1C:ERP export with complete HSE CAPA audit trail."""
    incidents = storage.get_incidents()
    
    # Filter by period if needed
    now = time.time()
    if period == "shift":
        incidents = incidents[-50:] if len(incidents) > 50 else incidents
    elif period == "7d":
        incidents = incidents[-150:] if len(incidents) > 150 else incidents
    elif period == "30d":
        incidents = incidents[-500:] if len(incidents) > 500 else incidents

    if format == "json":
        return {
            "report_generated_at": time.strftime("%d.%m.%Y %H:%M:%S"),
            "period": period,
            "total_incidents": len(incidents),
            "incidents": incidents
        }

    status_labels = {
        "new": "Новый (требует триажа)",
        "verified": "Подтверждено",
        "false_alarm": "Ложное срабатывание",
        "investigation": "В расследовании (CAPA)",
        "resolved": "Закрыто / Меры приняты"
    }

    # EXCEL XLSX FORMAT (with styled header, auto-fit column widths, borders)
    if format == "xlsx":
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Журнал ТБ и CAPA"
            ws.views.sheetView[0].showGridLines = True

            # Header styling
            header_fill = PatternFill(start_color="0284C7", end_color="0284C7", fill_type="solid")
            header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            border_thin = Border(
                left=Side(style='thin', color='CBD5E1'),
                right=Side(style='thin', color='CBD5E1'),
                top=Side(style='thin', color='CBD5E1'),
                bottom=Side(style='thin', color='CBD5E1')
            )
            align_center = Alignment(horizontal='center', vertical='center', wrap_text=True)
            align_left = Alignment(horizontal='left', vertical='center', wrap_text=True)

            headers = [
                "ID Инцидента", "Дата и время", "Камера / Локация", "Тип нарушения ТБ", 
                "Статус HSE", "Коренная причина (5 Почему)", "Мероприятие CAPA", 
                "Ответственный мастер", "Срок устранения", "История аудита"
            ]
            ws.append(headers)
            ws.row_dimensions[1].height = 32

            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = align_center
                cell.border = border_thin

            # Data rows
            zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
            white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

            for row_idx, inc in enumerate(incidents, start=2):
                inv = inc.get("investigation") or {}
                history = inc.get("history") or []
                hist_str = " -> ".join([f"[{h.get('time_str', '')}] {h.get('role', '')}: {h.get('status', '')}" for h in history])
                status_raw = inc.get("status", "new")
                status_human = status_labels.get(status_raw, status_raw)

                row_data = [
                    inc.get("id", f"INC-{row_idx}"),
                    inc.get("time_str", time.strftime("%H:%M:%S")),
                    inc.get("camera_title") or inc.get("camera_name", "ЦМОО • Шнековые сепараторы"),
                    ", ".join(inc.get("violations", ["Нарушение регламента СИЗ"])),
                    status_human,
                    inv.get("root_cause", "") or inv.get("why5", "") or inv.get("why1", "Требуется служебное расследование"),
                    inv.get("corrective_action", "") or "Инструктаж смены по безопасности",
                    inv.get("responsible_person", "") or "Мастер цеха ЦМОО",
                    inv.get("deadline", "") or time.strftime("%d.%m.%Y"),
                    hist_str or f"[{inc.get('time_str', '')}] Диспетчер: Зафиксировано"
                ]
                ws.append(row_data)
                ws.row_dimensions[row_idx].height = 24
                current_fill = zebra_fill if row_idx % 2 == 0 else white_fill

                for col_idx in range(1, len(row_data) + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.fill = current_fill
                    cell.border = border_thin
                    cell.font = Font(name="Calibri", size=10)
                    if col_idx in (1, 2, 5, 9):
                        cell.alignment = align_center
                    else:
                        cell.alignment = align_left

            # Explicit optimal column widths
            col_widths = {
                1: 16,  # ID
                2: 18,  # Дата/время
                3: 30,  # Камера/локация
                4: 34,  # Нарушения
                5: 22,  # Статус HSE
                6: 38,  # 5 Почему
                7: 42,  # Мероприятия CAPA
                8: 24,  # Ответственный
                9: 16,  # Срок
                10: 36  # Аудит
            }
            for col_idx, width in col_widths.items():
                col_letter = openpyxl.utils.get_column_letter(col_idx)
                ws.column_dimensions[col_letter].width = width

            output_xlsx = io.BytesIO()
            wb.save(output_xlsx)
            output_xlsx.seek(0)
            
            return Response(
                content=output_xlsx.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=rvk_hse_director_report_{time.strftime('%Y%m%d_%H%M')}.xlsx"}
            )
        except Exception as e:
            print("Fallback to CSV due to openpyxl error:", e)

    # CSV format for Excel and 1C:Enterprise (with sep=; and UTF-8 BOM for instant column parsing)
    output = io.StringIO()
    output.write("sep=;\n")  # Instructs Excel on Windows to split by semicolon
    writer = csv.writer(output, delimiter=';')
    writer.writerow([
        "ID Инцидента", "Дата и время", "Камера / Локация", "Тип нарушения", 
        "Статус HSE", "Коренная причина (5 Почему)", "Мероприятие CAPA", 
        "Ответственный мастер", "Срок исполнения", "История аудита"
    ])
    
    for inc in incidents:
        inv = inc.get("investigation") or {}
        history = inc.get("history") or []
        hist_str = " -> ".join([f"[{h.get('time_str', '')}] {h.get('role', '')}: {h.get('status', '')}" for h in history])
        status_raw = inc.get("status", "new")
        status_human = status_labels.get(status_raw, status_raw)
        
        writer.writerow([
            inc.get("id", ""),
            inc.get("time_str", ""),
            inc.get("camera_title") or inc.get("camera_name", "ЦМОО • Шнековые сепараторы"),
            ", ".join(inc.get("violations", [])),
            status_human,
            inv.get("root_cause", "") or inv.get("why5", "") or inv.get("why1", ""),
            inv.get("corrective_action", ""),
            inv.get("responsible_person", ""),
            inv.get("deadline", ""),
            hist_str
        ])
    
    csv_bytes = output.getvalue().encode('utf-8-sig')
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rvk_hse_director_report_{time.strftime('%Y%m%d_%H%M')}.csv"}
    )

@router.delete("/clear")
def clear_incidents():
    storage.save_incidents([])
    sys_logger.warning("AUDIT", "Оперативный журнал инцидентов ТБ полностью очищен администратором")
    return {"status": "cleared"}
