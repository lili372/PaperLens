"""选题建议系统 - 缓存优先总编排入口。

职责：串起搜索、PDF 下载、画像、候选方向、RAG、查证、审查、单次修改和报告生成。
这个入口沉淀探针中已经验证的编排逻辑，默认缓存优先。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arxiv

from src.agents.proposal_pipeline.cache_utils import ensure_run_dirs, get_proposal_cache_dir, now_iso
from src.agents.proposal_pipeline.candidate_review_batch_node import run_batch as run_candidate_review_batch
from src.agents.proposal_pipeline.pdf_download_node import (
    PDF_MAX_ANALYSIS_PAGES,
    PDF_SUCCESS_PER_QUERY_LIMIT,
    PDF_SKIP_TITLE_KEYWORDS,
    PDF_TARGET_SUCCESS_COUNT,
    download_pdfs_with_bucket_backfill,
)
from src.agents.proposal_pipeline.pdf_parse import available_pdf_papers, parse_pdfs_for_rag
from src.agents.proposal_pipeline.profile_node import run_profile
from src.agents.proposal_pipeline.report_node import build_markdown, build_report, write_json
from src.agents.proposal_pipeline.revision_batch_node import run_batch as run_revision_batch
from src.agents.proposal_pipeline.search_node import (
    BASE_MAX_RESULTS,
    MODULE_MAX_RESULTS,
    _merge_paper,
    build_base_raw_query,
    build_combo_raw_query,
)
from src.agents.proposal_pipeline.topic_node import run_topic
from src.core.config import config
from src.core.state_models import ProposalSearchPlan
from src.knowledge.knowledge import knowledge_base
from src.tasks.paper_search import PaperSearcher


PIPELINE_STEPS = [
    "search",
    "pdf_download",
    "profiles",
    "topic",
    "rag",
    "review_batch",
    "revision_batch",
    "report",
]

MIN_PROFILE_COUNT = 3
LOW_PROFILE_COUNT_LIMIT = 5
LOW_CANDIDATE_COUNT_LIMIT = 2


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def existing(path: Path) -> bool:
    """判断缓存产物是否存在且非空。"""
    return path.exists() and path.stat().st_size > 0


def count_profiles(profile_dir: Path) -> int:
    """统计已生成的画像文件数量。"""
    return len(profile_paths(profile_dir))


def validate_profile_count(profile_dir: Path) -> str | None:
    """检查画像数量是否足以继续生成候选方向。"""
    count = count_profiles(profile_dir)
    if count < MIN_PROFILE_COUNT:
        raise ValueError(f"论文画像成功数只有 {count} 篇，不足以生成可靠研究方向。请扩大检索范围或重新确认论文。")
    if count <= LOW_PROFILE_COUNT_LIMIT:
        return f"论文画像成功数为 {count} 篇，证据规模偏小，后续方向仅适合作为初步探索。"
    return None


def validate_candidate_count(candidates_path: Path) -> str | None:
    """检查候选方向数量。"""
    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("候选方向文件必须是 JSON 数组")
    count = len(candidates)
    if count == 0:
        raise ValueError("当前材料没有生成候选研究方向，请扩大检索范围或调整模块词。")
    if count <= LOW_CANDIDATE_COUNT_LIMIT:
        return f"候选研究方向只有 {count} 个，当前材料可组合空间偏少。"
    return None


def embedding_info() -> dict[str, Any]:
    """读取 embedding 配置。"""
    embedding = config.get("embedding-model")
    provider = config.get(embedding.get("model-provider"))
    return {
        "name": embedding.get("model"),
        "dimension": embedding.get("dimension"),
        "base_url": provider.get("base_url"),
        "api_key": provider.get("api_key"),
    }


def build_plan(args: argparse.Namespace) -> ProposalSearchPlan:
    """从命令行参数构建检索计划。"""
    return ProposalSearchPlan(
        base_query=args.base_query,
        module_terms=(args.module_terms or [])[:3],
        start_date=args.start_date,
        end_date=args.end_date,
        rationale=args.rationale,
    )


async def run_search(
    plan: ProposalSearchPlan,
    user_request: str,
    search_results_path: Path,
    force: bool,
) -> str:
    """执行 arXiv 搜索，输出原始去重结果。"""
    if existing(search_results_path) and not force:
        return "cache"
    if not plan.base_query:
        raise ValueError("缺少 base_query，无法搜索")

    searcher = PaperSearcher()
    paper_pool: dict[str, dict[str, Any]] = {}

    base_raw_query = build_base_raw_query(plan.base_query, plan.start_date, plan.end_date)
    base_results = await searcher.search_raw_query(
        raw_query=base_raw_query,
        max_results=BASE_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    for paper in base_results:
        _merge_paper(paper_pool, paper, "base", plan.base_query)

    for module_term in plan.module_terms:
        combo_raw_query = build_combo_raw_query(plan.base_query, module_term, plan.start_date, plan.end_date)
        module_results = await searcher.search_raw_query(
            raw_query=combo_raw_query,
            max_results=MODULE_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        for paper in module_results:
            _merge_paper(paper_pool, paper, "module", f"{plan.base_query} AND {module_term}")

    results = list(paper_pool.values())
    results.sort(key=lambda paper: (paper.get("published_date") or ""), reverse=True)
    if not results:
        raise ValueError("搜索结果为空，请调整检索计划")

    dump_json(
        search_results_path.parent / "search_results_all.json",
        {
            "created_at": now_iso(),
            "user_request": user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(results),
            "papers": results,
        },
    )
    dump_json(
        search_results_path,
        {
            "created_at": now_iso(),
            "user_request": user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(results),
            "selected_count": len(results),
            "selection_policy": {
                "type": "raw_dedup_before_pdf_backfill",
                "note": "PDF 下载阶段会按检索桶补位，并覆盖本文件为最终进入分析的成功 PDF 列表。",
                "buckets": [plan.base_query] + [f"{plan.base_query} AND {term}" for term in plan.module_terms],
            },
            "count": len(results),
            "papers": results,
        },
    )
    return "run"


async def run_pdf_download(
    plan: ProposalSearchPlan,
    user_request: str,
    search_results_path: Path,
    pdf_manifest_path: Path,
    pdf_dir: Path,
    force: bool,
) -> str:
    """下载 PDF，并按桶补位生成最终进入分析的论文列表。"""
    search_results = read_json(search_results_path)
    search_all_path = search_results_path.parent / "search_results_all.json"
    if existing(search_all_path):
        search_results = read_json(search_all_path)
    papers = list(search_results.get("papers") or [])
    if not papers:
        raise ValueError("没有可下载的搜索结果")

    target_count = min(PDF_TARGET_SUCCESS_COUNT, len([paper for paper in papers if paper.get("paper_id")]))
    if existing(pdf_manifest_path) and not force:
        cached_manifest = read_json(pdf_manifest_path)
        cached_policy = cached_manifest.get("selection_policy") or {}
        cached_target = cached_policy.get("target_success_count")
        cached_success = int(cached_manifest.get("success_count") or 0)
        cached_attempted = int(cached_manifest.get("attempted_count") or 0)
        cache_matches_target_policy = cached_target == target_count
        cache_already_exhausted = cached_attempted >= len(papers)
        if cache_matches_target_policy and (cached_success >= target_count or cache_already_exhausted):
            return "cache"

    download_result = await download_pdfs_with_bucket_backfill(papers, plan, pdf_dir)
    selected_papers = download_result["selected_papers"]
    manifest = {
        "created_at": now_iso(),
        "user_request": user_request,
        "search_plan": plan.model_dump(),
        "run_dir": str(pdf_manifest_path.parent),
        "pdf_dir": str(pdf_dir),
        "total": len(selected_papers),
        "attempted_count": download_result["attempted_count"],
        "success_count": download_result["success_count"],
        "failed_count": download_result["failed_count"],
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
        "attempted_papers": download_result["attempted_papers"],
    }
    dump_json(pdf_manifest_path, manifest)
    dump_json(
        search_results_path,
        {
            "created_at": now_iso(),
            "user_request": user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(papers),
            "selected_count": len(selected_papers),
            "selection_policy": manifest["selection_policy"],
            "bucket_reports": download_result["bucket_reports"],
            "fill_report": download_result["fill_report"],
            "count": len(selected_papers),
            "papers": selected_papers,
        },
    )
    if download_result["success_count"] == 0:
        raise ValueError("PDF 全部下载失败，无法继续")
    return "run"


async def run_profiles(
    pdf_manifest_path: Path,
    profile_dir: Path,
    force: bool,
    max_papers: int | None,
) -> str:
    """为已下载 PDF 生成论文画像。"""
    manifest = read_json(pdf_manifest_path)
    papers = available_pdf_papers(manifest)
    if max_papers:
        papers = papers[:max_papers]
    profile_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, Path]] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id"))
        out_path = profile_dir / f"paper_profile_{paper_id}.md"
        if existing(out_path) and not force:
            continue
        pending.append((paper_id, out_path))

    if not pending:
        return "cache"

    concurrency = max(1, config.get_int("proposal_profile_concurrency", 3))
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(paper_id: str, out_path: Path) -> None:
        async with semaphore:
            await run_profile(
                manifest_path=pdf_manifest_path,
                paper_id=paper_id,
                out_path=out_path,
                max_chars_per_section=3500,
            )

    await asyncio.gather(*(run_one(paper_id, out_path) for paper_id, out_path in pending))
    return "run"


def profile_paths(profile_dir: Path) -> list[Path]:
    """列出画像文件。"""
    return sorted(profile_dir.glob("paper_profile_*.md"))


def topic_input_profile_paths(topic_debug_path: Path) -> list[Path]:
    """从候选生成调试文件读取当时实际使用的画像路径。"""
    if not topic_debug_path.exists():
        return []
    content = topic_debug_path.read_text(encoding="utf-8")
    paths: list[Path] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        text = line[2:].strip()
        if "paper_profile_" not in text:
            continue
        paths.append(Path(text))
    return paths


def selected_profile_paths(profile_dir: Path, candidates_path: Path) -> list[Path]:
    """优先复用候选生成时的画像输入顺序。"""
    cached_inputs = topic_input_profile_paths(candidates_path.with_suffix(".md"))
    existing_inputs = [path for path in cached_inputs if path.exists()]
    if existing_inputs:
        return existing_inputs
    return profile_paths(profile_dir)


def needed_profile_ids_from_topic_debug(topic_debug_path: Path) -> list[str]:
    """从候选生成调试文件提取画像 paper_id。"""
    ids: list[str] = []
    for path in topic_input_profile_paths(topic_debug_path):
        match = re.search(r"paper_profile_(.+)\.md$", path.name)
        if match:
            ids.append(match.group(1))
    return ids


async def run_topics(profile_dir: Path, candidates_path: Path, force: bool) -> str:
    """生成候选研究方向。"""
    if existing(candidates_path) and not force:
        return "cache"
    profiles = profile_paths(profile_dir)
    if not profiles:
        raise ValueError("没有论文画像，无法生成候选方向")
    await run_topic(profile_paths=profiles, out_path=candidates_path)
    return "run"


async def run_rag_build(pdf_manifest_path: Path, rag_manifest_path: Path, force: bool) -> tuple[str, str]:
    """解析 PDF 并创建 Chroma 临时知识库。"""
    if existing(rag_manifest_path) and not force:
        rag_manifest = read_json(rag_manifest_path)
        db_id = str(rag_manifest.get("db_id") or "")
        if db_id:
            return "cache", db_id

    manifest = read_json(pdf_manifest_path)
    papers = available_pdf_papers(manifest)
    documents, metadatas, ids, parse_reports, dedupe_summary = parse_pdfs_for_rag(
        papers,
        max_chars=3500,
        overlap=300,
    )
    database_info = await knowledge_base.create_database(
        f"V2 PDF RAG {pdf_manifest_path.parent.name}",
        "V2 选题流水线临时 PDF chunk 知识库，用于候选方向证据召回。",
        kb_type=config.get("KB_TYPE", "chroma"),
        embed_info=embedding_info(),
        llm_info=None,
    )
    db_id = database_info["db_id"]
    config.set("tmp_db_id", db_id)
    batch_size = 64
    for start in range(0, len(documents), batch_size):
        await knowledge_base.add_processed_content(
            db_id,
            {
                "documents": documents[start: start + batch_size],
                "metadatas": metadatas[start: start + batch_size],
                "ids": ids[start: start + batch_size],
            },
        )

    dump_json(
        rag_manifest_path,
        {
            "db_id": db_id,
            "kb_type": config.get("KB_TYPE", "chroma"),
            "source_manifest": str(pdf_manifest_path),
            "paper_count": len(papers),
            "chunk_count": len(documents),
            "chunk_chars": 3500,
            "chunk_overlap": 300,
            "dedupe_rule": "paper_id + section_key + normalized_text_sha256",
            "dedupe_summary": dedupe_summary,
            "papers": parse_reports,
        },
    )
    return "run", db_id


async def run_review_batch(
    db_id: str,
    pdf_manifest_path: Path,
    candidates_path: Path,
    profile_dir: Path,
    out_dir: Path,
    force: bool,
) -> str:
    """批量执行 RAG 召回、查证和审查。"""
    summary_path = out_dir / "summary.json"
    if existing(summary_path) and not force:
        return "cache"
    args = SimpleNamespace(
        existing_db_id=db_id,
        manifest=str(pdf_manifest_path),
        candidates=str(candidates_path),
        profiles=[str(path) for path in selected_profile_paths(profile_dir, candidates_path)],
        candidate_ids=None,
        out_dir=str(out_dir),
        top_k_per_bucket=4,
        similarity_threshold=0.0,
        max_chars_per_result=4500,
    )
    await run_candidate_review_batch(args)
    return "run"


async def run_revisions(candidates_path: Path, review_dir: Path, revision_dir: Path, force: bool) -> str:
    """按审查结论执行单次修改和分流。"""
    final_path = revision_dir / "final_candidates_probe.json"
    if existing(final_path) and not force:
        return "cache"
    args = SimpleNamespace(
        candidates=str(candidates_path),
        review_summary=str(review_dir / "summary.json"),
        review_dir=str(review_dir),
        out_dir=str(revision_dir),
    )
    await run_revision_batch(args)
    return "run"


def run_report(
    search_results_path: Path,
    pdf_manifest_path: Path,
    revision_dir: Path,
    review_dir: Path,
    report_md_path: Path,
    report_json_path: Path,
    force: bool,
) -> str:
    """拼装最终中文报告。"""
    if existing(report_md_path) and existing(report_json_path) and not force:
        return "cache"
    args = SimpleNamespace(
        search_results=str(search_results_path),
        pdf_manifest=str(pdf_manifest_path),
        final_candidates=str(revision_dir / "final_candidates_probe.json"),
        deferred_candidates=str(revision_dir / "deferred_candidates_probe.json"),
        review_summary=str(review_dir / "summary.json"),
    )
    report = build_report(args)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(build_markdown(report), encoding="utf-8")
    write_json(report_json_path, report)
    return "run"


async def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """运行 V2 选题建议全链路。"""
    plan = build_plan(args)
    run_dir = Path(args.run_dir) if args.run_dir else get_proposal_cache_dir(plan)
    paths = ensure_run_dirs(run_dir)
    profile_dir = run_dir / "profiles"
    candidates_path = run_dir / "topic_proposals_probe_json.json"
    rag_manifest_path = run_dir / "rag_manifest.json"
    review_dir = run_dir / "candidate_review_batch_claim_buckets"
    revision_dir = run_dir / "revision_batch_probe"
    report_md_path = run_dir / "final_report_probe.md"
    report_json_path = run_dir / "final_report_probe.json"
    step_records: list[dict[str, str]] = []

    force_steps = set(args.force_steps or [])

    async def record(name: str, result: str) -> None:
        step_records.append({"step": name, "mode": result})
        print(f"[{name}] {result}")

    await record(
        "search",
        await run_search(plan, args.user_request, paths["search_results"], "search" in force_steps),
    )
    await record(
        "pdf_download",
        await run_pdf_download(
            plan,
            args.user_request,
            paths["search_results"],
            paths["pdf_manifest"],
            paths["pdf_dir"],
            "pdf_download" in force_steps,
        ),
    )
    topic_exists = existing(candidates_path)
    profile_force = "profiles" in force_steps
    if topic_exists and not profile_force:
        profile_mode = "cache"
    else:
        profile_mode = await run_profiles(paths["pdf_manifest"], profile_dir, profile_force, args.max_profile_papers)
    await record("profiles", profile_mode)
    profile_warning = validate_profile_count(profile_dir)
    if profile_warning:
        print(f"[profiles] warning: {profile_warning}")
    await record("topic", await run_topics(profile_dir, candidates_path, "topic" in force_steps))
    candidate_warning = validate_candidate_count(candidates_path)
    if candidate_warning:
        print(f"[topic] warning: {candidate_warning}")
    rag_mode, db_id = await run_rag_build(paths["pdf_manifest"], rag_manifest_path, "rag" in force_steps)
    await record("rag", rag_mode)
    await record(
        "review_batch",
        await run_review_batch(
            db_id,
            paths["pdf_manifest"],
            candidates_path,
            profile_dir,
            review_dir,
            "review_batch" in force_steps,
        ),
    )
    await record(
        "revision_batch",
        await run_revisions(candidates_path, review_dir, revision_dir, "revision_batch" in force_steps),
    )
    await record(
        "report",
        run_report(
            paths["search_results"],
            paths["pdf_manifest"],
            revision_dir,
            review_dir,
            report_md_path,
            report_json_path,
            "report" in force_steps,
        ),
    )

    manifest = {
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "search_plan": plan.model_dump(),
        "steps": step_records,
        "outputs": {
            "search_results": str(paths["search_results"]),
            "pdf_manifest": str(paths["pdf_manifest"]),
            "profiles_dir": str(profile_dir),
            "topic_candidates": str(candidates_path),
            "rag_manifest": str(rag_manifest_path),
            "review_summary": str(review_dir / "summary.json"),
            "revision_final_candidates": str(revision_dir / "final_candidates_probe.json"),
            "final_report_md": str(report_md_path),
            "final_report_json": str(report_json_path),
        },
        "profile_inputs_for_candidates": [
            str(path) for path in selected_profile_paths(profile_dir, candidates_path)
        ],
    }
    dump_json(run_dir / "pipeline_manifest.json", manifest)
    return manifest
