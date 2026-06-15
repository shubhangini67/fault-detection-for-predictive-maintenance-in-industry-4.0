"""Rule-based maintenance recommendation engine for detected equipment faults."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MaintenancePriority(str, Enum):
    ROUTINE = "ROUTINE"
    SCHEDULED = "SCHEDULED"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class MaintenanceAction(str, Enum):
    MONITOR = "MONITOR"
    INSPECT = "INSPECT"
    LUBRICATE = "LUBRICATE"
    ALIGN = "ALIGN"
    REPLACE_BEARING = "REPLACE_BEARING"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class MaintenanceRecommendation:
    priority: MaintenancePriority
    actions: list[MaintenanceAction]
    summary: str
    technical_notes: list[str] = field(default_factory=list)
    estimated_downtime_hours: float = 0.0
    spare_parts: list[str] = field(default_factory=list)
    cmms_work_order_title: str = ""


_PLAYBOOKS: dict[str, dict] = {
    "NB": {
        "priority": MaintenancePriority.ROUTINE,
        "actions": [MaintenanceAction.MONITOR],
        "summary": "Equipment operating within baseline vibration envelope.",
        "technical_notes": [
            "Continue scheduled condition monitoring at current interval.",
            "No immediate mechanical intervention required.",
        ],
        "estimated_downtime_hours": 0.0,
        "spare_parts": [],
    },
    "IR - 7": {
        "priority": MaintenancePriority.SCHEDULED,
        "actions": [MaintenanceAction.INSPECT, MaintenanceAction.LUBRICATE, MaintenanceAction.MONITOR],
        "summary": "Early-stage inner race defect detected — schedule targeted inspection within 72 hours.",
        "technical_notes": [
            "Inspect inner race track for spalling or micro-pitting on drive-end bearing.",
            "Verify lubrication viscosity and contamination levels (ISO 4406).",
            "Increase vibration sampling rate to 1-minute intervals until next planned stop.",
            "Compare DE vs FE amplitude ratio; rising DE dominance confirms inner race progression.",
        ],
        "estimated_downtime_hours": 2.0,
        "spare_parts": ["Bearing grease (OEM spec)", "Inner race inspection kit"],
    },
    "IR - 21": {
        "priority": MaintenancePriority.CRITICAL,
        "actions": [MaintenanceAction.SHUTDOWN, MaintenanceAction.REPLACE_BEARING],
        "summary": "Critical inner race fault — initiate controlled shutdown and bearing replacement.",
        "technical_notes": [
            "Stop asset under controlled ramp-down to prevent secondary damage to shaft and housing.",
            "Replace drive-end bearing assembly; inspect shaft for fretting corrosion.",
            "Perform laser shaft alignment post replacement (tolerance ≤ 0.05 mm).",
            "Root-cause analysis: check for misalignment, unbalance, or inadequate lubrication.",
        ],
        "estimated_downtime_hours": 8.0,
        "spare_parts": [
            "Drive-end bearing (OEM part)",
            "Shaft seal kit",
            "Alignment shims",
        ],
    },
    "OR - 7": {
        "priority": MaintenancePriority.SCHEDULED,
        "actions": [MaintenanceAction.INSPECT, MaintenanceAction.ALIGN, MaintenanceAction.MONITOR],
        "summary": "Early outer race anomaly — plan maintenance window within 5–7 operating days.",
        "technical_notes": [
            "Inspect outer race housing fit and mounting bolt torque.",
            "Check for soft foot and foundation looseness contributing to outer race loading.",
            "Schedule thermography scan during next production break.",
        ],
        "estimated_downtime_hours": 3.0,
        "spare_parts": ["Mounting hardware kit", "Housing gasket"],
    },
    "OR - 21": {
        "priority": MaintenancePriority.URGENT,
        "actions": [MaintenanceAction.SHUTDOWN, MaintenanceAction.REPLACE_BEARING, MaintenanceAction.ALIGN],
        "summary": "Severe outer race defect — reduce load and replace bearing at earliest safe window.",
        "technical_notes": [
            "Derate motor load by 30% until replacement if immediate shutdown is not feasible.",
            "Replace outer race bearing and inspect housing bore for damage.",
            "Mandatory precision alignment and balancing verification before return to service.",
        ],
        "estimated_downtime_hours": 6.0,
        "spare_parts": [
            "Outer race bearing (OEM part)",
            "Housing repair sleeve (if bore scored)",
            "Coupling element",
        ],
    },
}


def recommend_maintenance(
    scenario_key: str,
    fault_detected: bool,
    confidence: float = 1.0,
    model_name: str = "edge_classifier",
) -> MaintenanceRecommendation:
    if not fault_detected:
        playbook = _PLAYBOOKS["NB"]
        wo_title = "PM-ROUTINE: Continue predictive monitoring"
    else:
        playbook = _PLAYBOOKS.get(scenario_key, _PLAYBOOKS["IR - 7"])
        wo_title = f"PM-{playbook['priority'].value}: {playbook['summary'][:60]}"

    notes = list(playbook["technical_notes"])
    if confidence < 0.75:
        notes.insert(
            0,
            f"Edge inference confidence {confidence:.0%} below threshold — "
            "recommend secondary manual inspection before major intervention.",
        )
    notes.append(f"Alert source: {model_name} (on-device Edge AI inference).")

    return MaintenanceRecommendation(
        priority=playbook["priority"],
        actions=playbook["actions"],
        summary=playbook["summary"],
        technical_notes=notes,
        estimated_downtime_hours=playbook["estimated_downtime_hours"],
        spare_parts=playbook["spare_parts"],
        cmms_work_order_title=wo_title,
    )


def recommendation_to_dict(rec: MaintenanceRecommendation) -> dict:
    return {
        "priority": rec.priority.value,
        "actions": [a.value for a in rec.actions],
        "summary": rec.summary,
        "technical_notes": rec.technical_notes,
        "estimated_downtime_hours": rec.estimated_downtime_hours,
        "spare_parts": rec.spare_parts,
        "cmms_work_order_title": rec.cmms_work_order_title,
    }
