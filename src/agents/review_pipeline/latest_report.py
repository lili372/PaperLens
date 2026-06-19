"""V1 综述最新报告保存与读取。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.config import config


def review_runs_root() -> Path:
    return Path(config.get("SAVE_DIR")) / "review_runs"


def save_review_report(current_state: Any) -> dict[str, str] | None:
    """保存一次 V1 综述结果，供前端“查看最新报告”读取。"""
    markdown = getattr(current_state, "review_markdown", None)
    if not markdown:
        return None

    root = review_runs_root()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    extracted = getattr(current_state, "extracted_data", None)
    paper_count = len(getattr(extracted, "papers", []) or []) if extracted is not None else 0
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_request": getattr(current_state, "user_request", ""),
        "paper_count": paper_count,
        "review_warning": getattr(current_state, "review_warning", None),
        "faithfulness_report": getattr(current_state, "faithfulness_report", None),
    }

    md_path = run_dir / "review_report.md"
    json_path = run_dir / "review_report.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "markdown": str(md_path), "json": str(json_path)}


def load_latest_review_report() -> dict[str, Any] | None:
    """读取最近一次 V1 综述报告。"""
    root = review_runs_root()
    report_paths = sorted(
        root.glob("*/review_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not report_paths:
        return None

    json_path = report_paths[0]
    md_path = json_path.with_suffix(".md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(json_path.parent),
        "report": payload,
        "markdown": md_path.read_text(encoding="utf-8") if md_path.exists() else "",
    }
