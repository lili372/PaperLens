"""选题建议系统的本地缓存工具。"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.core.config import config
from src.core.state_models import ProposalSearchPlan
from src.utils import hashstr


def _safe_name(text: str, max_length: int = 60) -> str:
    """将检索词转成适合目录名的字符串。"""
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", text.strip())
    name = name.strip("._-")
    if not name:
        return "proposal"
    return name[:max_length]


def get_proposal_cache_dir(plan: ProposalSearchPlan) -> Path:
    """根据检索计划生成稳定缓存目录。"""
    payload = {
        "base_query": plan.base_query,
        "module_terms": plan.module_terms,
        "start_date": plan.start_date,
        "end_date": plan.end_date,
    }
    digest = hashstr(json.dumps(payload, ensure_ascii=False, sort_keys=True), length=10)
    base_name = _safe_name(plan.base_query or "proposal")
    return Path(config.get("SAVE_DIR")) / "proposal_runs" / f"{base_name}_{digest}"


def ensure_run_dirs(run_dir: Path) -> Dict[str, Path]:
    """创建一次 V2 调研运行需要的缓存目录。"""
    pdf_dir = run_dir / "pdfs"
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "pdf_dir": pdf_dir,
        "search_results": run_dir / "search_results.json",
        "pdf_manifest": run_dir / "pdf_manifest.json",
    }


def write_json(path: Path, data: Any) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    """当前时间 ISO 字符串。"""
    return datetime.now().isoformat(timespec="seconds")
