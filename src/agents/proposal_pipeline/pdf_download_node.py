"""选题建议系统 - PDF 下载节点。

职责：接收已确认的论文池，下载可用 PDF，并把本地路径写回论文元数据。
"""
import asyncio
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import fitz

from src.agents.proposal_pipeline.cache_utils import ensure_run_dirs, get_proposal_cache_dir, now_iso, write_json
from src.core.config import config
from src.core.state_models import BackToFrontData, ExecutionState, ProposalSearchPlan, State
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

PDF_TIMEOUT_SECONDS = config.get_int("proposal_pdf_timeout_seconds", 10)
PDF_MAX_RETRIES = config.get_int("proposal_pdf_max_retries", 1)
PDF_RETRY_DELAY_SECONDS = config.get_float("proposal_pdf_retry_delay_seconds", 1.0)
PDF_CONCURRENCY = config.get_int("proposal_pdf_concurrency", 3)
PDF_SUCCESS_PER_QUERY_LIMIT = config.get_int("proposal_pdf_success_per_query_limit", 3)
PDF_TARGET_SUCCESS_COUNT = config.get_int("proposal_pdf_target_success_count", 9)
PDF_REUSE_CROSS_RUN_CACHE = config.get_bool("proposal_pdf_reuse_cross_run_cache", False)
PDF_MAX_ANALYSIS_PAGES = config.get_int("proposal_pdf_max_analysis_pages", 25)
PDF_SKIP_TITLE_KEYWORDS = ["survey", "review", "tutorial", "overview", "roadmap"]


def _safe_filename(text: str, max_length: int = 90) -> str:
    """生成安全文件名片段。"""
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", text.strip())
    name = name.strip("._-")
    return (name or "paper")[:max_length]


def _paper_pdf_path(pdf_dir: Path, paper: Dict[str, Any]) -> Path:
    """根据论文元数据生成 PDF 路径。"""
    paper_id = _safe_filename(str(paper.get("paper_id") or "unknown"))
    title = _safe_filename(str(paper.get("title") or "paper"), max_length=70)
    return pdf_dir / f"{paper_id}_{title}.pdf"


def _find_cached_pdf(paper: Dict[str, Any], target_path: Path) -> Optional[Path]:
    """跨运行目录查找已下载过的同一篇 arXiv PDF。"""
    paper_id = _safe_filename(str(paper.get("paper_id") or ""))
    if not paper_id:
        return None
    proposal_root = Path(config.get("SAVE_DIR")) / "proposal_runs"
    if not proposal_root.exists():
        return None
    for pdf_path in proposal_root.glob(f"*/pdfs/{paper_id}_*.pdf"):
        if pdf_path.resolve() == target_path.resolve():
            continue
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path
    return None


def _pdf_page_count(pdf_path: Path) -> int:
    """读取 PDF 页数。"""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _analysis_skip_reason(paper: Dict[str, Any]) -> Optional[str]:
    """判断下载成功的论文是否应跳过分析。"""
    title = str(paper.get("title") or "").lower()
    for keyword in PDF_SKIP_TITLE_KEYWORDS:
        if keyword in title:
            return f"标题包含综述类关键词: {keyword}"

    pdf_path_text = paper.get("pdf_path")
    if pdf_path_text:
        try:
            page_count = _pdf_page_count(Path(pdf_path_text))
        except Exception as exc:
            logger.warning("读取 PDF 页数失败 paper_id=%s error=%s", paper.get("paper_id"), exc)
            return None
        paper["pdf_page_count"] = page_count
        if page_count > PDF_MAX_ANALYSIS_PAGES:
            return f"PDF 页数 {page_count} 超过上限 {PDF_MAX_ANALYSIS_PAGES}"
    return None


def _is_analysis_ready(paper: Dict[str, Any]) -> bool:
    """判断论文是否下载成功且适合进入分析。"""
    if paper.get("pdf_status") not in ("downloaded", "cached"):
        return False
    skip_reason = _analysis_skip_reason(paper)
    if skip_reason:
        paper["pdf_status"] = "skipped"
        paper["pdf_error"] = None
        paper["skip_reason"] = skip_reason
        return False
    return True


async def _download_one_pdf(
    session: aiohttp.ClientSession,
    paper: Dict[str, Any],
    pdf_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """下载单篇 PDF，并返回更新后的论文元数据。"""
    updated = dict(paper)
    pdf_url = updated.get("pdf_url")
    pdf_path = _paper_pdf_path(pdf_dir, updated)
    updated["pdf_path"] = str(pdf_path)

    if not pdf_url:
        updated["pdf_status"] = "failed"
        updated["pdf_error"] = "缺少 pdf_url"
        return updated

    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        updated["pdf_status"] = "cached"
        updated["pdf_error"] = None
        return updated

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cached_pdf = _find_cached_pdf(updated, pdf_path) if PDF_REUSE_CROSS_RUN_CACHE else None
    if cached_pdf:
        shutil.copy2(cached_pdf, pdf_path)
        updated["pdf_status"] = "cached"
        updated["pdf_error"] = None
        updated["pdf_cache_source"] = str(cached_pdf)
        return updated

    async with semaphore:
        last_error: Optional[str] = None
        for attempt in range(PDF_MAX_RETRIES + 1):
            try:
                async with session.get(pdf_url, timeout=PDF_TIMEOUT_SECONDS) as response:
                    if response.status != 200:
                        last_error = f"HTTP {response.status}"
                    else:
                        content = await response.read()
                        if not content.startswith(b"%PDF"):
                            last_error = "响应内容不是 PDF"
                        else:
                            pdf_path.write_bytes(content)
                            updated["pdf_status"] = "downloaded"
                            updated["pdf_error"] = None
                            return updated
            except Exception as exc:
                exc_text = str(exc).strip()
                last_error = f"{exc.__class__.__name__}: {exc_text}" if exc_text else exc.__class__.__name__

            if attempt < PDF_MAX_RETRIES:
                await asyncio.sleep(PDF_RETRY_DELAY_SECONDS)

        updated["pdf_status"] = "failed"
        updated["pdf_error"] = last_error or "未知下载错误"
        return updated


def _query_buckets(plan: ProposalSearchPlan) -> List[Dict[str, Any]]:
    """生成 PDF 补位下载使用的检索桶。"""
    buckets: List[Dict[str, Any]] = []
    if plan.base_query:
        buckets.append({"bucket": "base", "query": plan.base_query})
    for index, module_term in enumerate((plan.module_terms or [])[:3], start=1):
        buckets.append(
            {
                "bucket": f"module_{index}",
                "query": f"{plan.base_query} AND {module_term}",
                "module_term": module_term,
            }
        )
    return buckets


async def download_pdfs_with_bucket_backfill(
    papers: List[Dict[str, Any]],
    plan: ProposalSearchPlan,
    pdf_dir: Path,
) -> Dict[str, Any]:
    """按检索桶下载 PDF，并在失败时从同桶后续候选补位。

    第一阶段每个桶最多保留 3 篇下载成功的唯一论文，保证主题覆盖；
    第二阶段从原始去重论文池继续补位，尽量把最终分析集补到 9 篇。
    """
    semaphore = asyncio.Semaphore(PDF_CONCURRENCY)
    headers = {"User-Agent": "PaperLens/0.1 PDF downloader"}
    selected_ids = set()
    attempted: Dict[str, Dict[str, Any]] = {}
    selected_papers: List[Dict[str, Any]] = []
    bucket_reports: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(headers=headers) as session:
        for bucket in _query_buckets(plan):
            query = bucket["query"]
            candidates = [
                paper
                for paper in papers
                if query in (paper.get("matched_queries") or [])
            ]
            success_ids: List[str] = []
            failed_ids: List[str] = []
            skipped_ids: List[str] = []
            skipped_duplicate_ids: List[str] = []

            for paper in candidates:
                paper_id = str(paper.get("paper_id") or "")
                if not paper_id:
                    continue
                if paper_id in selected_ids:
                    skipped_duplicate_ids.append(paper_id)
                    continue

                updated = attempted.get(paper_id)
                if updated is None:
                    updated = await _download_one_pdf(session, paper, pdf_dir, semaphore)
                    attempted[paper_id] = updated

                if _is_analysis_ready(updated):
                    selected_ids.add(paper_id)
                    selected_papers.append(updated)
                    success_ids.append(paper_id)
                    if len(success_ids) >= PDF_SUCCESS_PER_QUERY_LIMIT:
                        break
                elif updated.get("pdf_status") == "skipped":
                    skipped_ids.append(paper_id)
                else:
                    failed_ids.append(paper_id)

            bucket_reports.append(
                {
                    **bucket,
                    "target_success_count": PDF_SUCCESS_PER_QUERY_LIMIT,
                    "candidate_count": len(candidates),
                    "success_count": len(success_ids),
                    "failed_count": len(failed_ids),
                    "skipped_count": len(skipped_ids),
                    "skipped_duplicate_count": len(skipped_duplicate_ids),
                    "exhausted": len(success_ids) < PDF_SUCCESS_PER_QUERY_LIMIT,
                    "success_paper_ids": success_ids,
                    "failed_paper_ids": failed_ids,
                    "skipped_paper_ids": skipped_ids,
                    "skipped_duplicate_paper_ids": skipped_duplicate_ids,
                }
            )

        target_success_count = min(PDF_TARGET_SUCCESS_COUNT, len([paper for paper in papers if paper.get("paper_id")]))
        fill_success_ids: List[str] = []
        fill_failed_ids: List[str] = []
        fill_skipped_ids: List[str] = []
        fill_skipped_duplicate_ids: List[str] = []
        fill_skipped_no_id_count = 0

        if len(selected_papers) < target_success_count:
            for paper in papers:
                paper_id = str(paper.get("paper_id") or "")
                if not paper_id:
                    fill_skipped_no_id_count += 1
                    continue
                if paper_id in selected_ids:
                    fill_skipped_duplicate_ids.append(paper_id)
                    continue

                updated = attempted.get(paper_id)
                if updated is None:
                    updated = await _download_one_pdf(session, paper, pdf_dir, semaphore)
                    attempted[paper_id] = updated

                if _is_analysis_ready(updated):
                    selected_ids.add(paper_id)
                    selected_papers.append(updated)
                    fill_success_ids.append(paper_id)
                    if len(selected_papers) >= target_success_count:
                        break
                elif updated.get("pdf_status") == "skipped":
                    fill_skipped_ids.append(paper_id)
                else:
                    fill_failed_ids.append(paper_id)

        fill_report = {
            "bucket": "target_fill",
            "query": "原始去重论文池补位",
            "target_success_count": target_success_count,
            "success_count": len(fill_success_ids),
            "failed_count": len(fill_failed_ids),
            "skipped_count": len(fill_skipped_ids),
            "skipped_duplicate_count": len(fill_skipped_duplicate_ids),
            "skipped_no_id_count": fill_skipped_no_id_count,
            "exhausted": len(selected_papers) < target_success_count,
            "success_paper_ids": fill_success_ids,
            "failed_paper_ids": fill_failed_ids,
            "skipped_paper_ids": fill_skipped_ids,
            "skipped_duplicate_paper_ids": fill_skipped_duplicate_ids,
        }

    attempted_papers = list(attempted.values())
    return {
        "selected_papers": selected_papers,
        "attempted_papers": attempted_papers,
        "bucket_reports": bucket_reports,
        "fill_report": fill_report,
        "target_success_count": target_success_count,
        "success_count": len(selected_papers),
        "failed_count": sum(1 for paper in attempted_papers if paper.get("pdf_status") == "failed"),
        "skipped_count": sum(1 for paper in attempted_papers if paper.get("pdf_status") == "skipped"),
        "attempted_count": len(attempted_papers),
    }


def _fallback_plan_from_state(state: State) -> ProposalSearchPlan:
    """缺少 proposal_search_plan 时，用搜索结果构造一个兜底缓存目录计划。"""
    current_state = state["value"]
    return ProposalSearchPlan(
        base_query=current_state.user_request,
        module_terms=[],
        start_date=None,
        end_date=None,
        rationale="由 PDF 下载节点兜底生成，仅用于缓存目录",
    )


async def proposal_pdf_download_node(state: State) -> State:
    """下载 V2 搜索结果中的 PDF，并写入 manifest。"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="initializing", data=None))

    papers: List[Dict[str, Any]] = list(current_state.search_results or [])
    if not papers:
        msg = "PDF 下载失败：没有可下载的论文搜索结果"
        current_state.error.reading_node_error = msg
        await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="error", data=msg))
        return {"value": current_state}

    plan = current_state.proposal_search_plan or _fallback_plan_from_state(state)
    run_dir = Path(current_state.proposal_run_dir) if current_state.proposal_run_dir else get_proposal_cache_dir(plan)
    paths = ensure_run_dirs(run_dir)
    current_state.proposal_run_dir = str(run_dir)
    current_state.pdf_manifest_path = str(paths["pdf_manifest"])

    await state_queue.put(BackToFrontData(
        step=ExecutionState.READING,
        state="thinking",
        data=f"正在下载 {len(papers)} 篇论文 PDF...\n",
    ))

    try:
        download_result = await download_pdfs_with_bucket_backfill(papers, plan, paths["pdf_dir"])
        selected_papers = download_result["selected_papers"]
        attempted_papers = download_result["attempted_papers"]
        success_count = download_result["success_count"]
        failed_count = download_result["failed_count"]
        current_state.search_results = selected_papers

        manifest = {
            "created_at": now_iso(),
            "user_request": current_state.user_request,
            "search_plan": plan.model_dump(),
            "run_dir": str(run_dir),
            "pdf_dir": str(paths["pdf_dir"]),
            "total": len(selected_papers),
            "attempted_count": download_result["attempted_count"],
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": download_result["skipped_count"],
            "selection_policy": {
                "type": "pdf_success_per_query_bucket",
                "success_per_query_limit": PDF_SUCCESS_PER_QUERY_LIMIT,
                "target_success_count": download_result["target_success_count"],
                "max_analysis_pages": PDF_MAX_ANALYSIS_PAGES,
                "skip_title_keywords": PDF_SKIP_TITLE_KEYWORDS,
            },
            "bucket_reports": download_result["bucket_reports"],
            "fill_report": download_result["fill_report"],
            "papers": selected_papers,
            "attempted_papers": attempted_papers,
        }
        write_json(paths["pdf_manifest"], manifest)
        write_json(paths["search_results"], {
            "created_at": now_iso(),
            "user_request": current_state.user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(papers),
            "selected_count": len(selected_papers),
            "selection_policy": manifest["selection_policy"],
            "bucket_reports": download_result["bucket_reports"],
            "fill_report": download_result["fill_report"],
            "count": len(selected_papers),
            "papers": selected_papers,
        })

        state_text = "completed" if success_count > 0 else "error"
        message = (
            f"PDF 下载完成，进入分析 {success_count} 篇，失败尝试 {failed_count} 篇，"
            f"manifest 已保存到 {paths['pdf_manifest']}"
        )
        if success_count == 0:
            current_state.error.reading_node_error = message

        await state_queue.put(BackToFrontData(step=ExecutionState.READING, state=state_text, data=message))
        return {"value": current_state}

    except Exception as exc:
        err_msg = f"PDF 下载节点失败: {str(exc)}"
        logger.error(err_msg)
        current_state.error.reading_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="error", data=err_msg))
        return {"value": current_state}
