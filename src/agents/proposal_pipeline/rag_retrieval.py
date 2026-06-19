"""选题建议系统 - 候选方向 RAG 召回工具。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.knowledge.knowledge import knowledge_base


BUCKET_SECTION_PRIORITIES = {
    "base_problem": ["abstract", "introduction", "conclusion", "related_work", "background"],
    "module_method": ["abstract", "method", "conclusion"],
    "connection": ["method", "experiment", "discussion", "conclusion", "background"],
}

BUCKET_LABELS = {
    "base_problem": "base 论文的问题、局限或可改进空间",
    "module_method": "module 论文提供的可迁移方法模块",
    "connection": "module 的输入、输出、依赖条件和 base 问题之间的连接可行性",
}

TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "under",
    "via",
    "with",
    "partial",
    "multi",
    "multi-label",
    "label",
    "learning",
    "pml",
}


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def title_tokens(title: str) -> set[str]:
    return {
        token
        for token in normalize_for_match(title).split()
        if len(token) >= 3 and token not in TITLE_STOPWORDS
    }


def build_profile_order_map(profile_paths: list[str | Path] | None) -> dict[int, str]:
    """把候选生成时的“论文N”映射到画像文件里的 paper_id。"""
    if not profile_paths:
        return {}

    mapping: dict[int, str] = {}
    for index, profile_path in enumerate(profile_paths, start=1):
        path = Path(profile_path)
        match = re.match(r"paper_profile_(.+)\.md$", path.name)
        if match:
            mapping[index] = match.group(1)
    return mapping


def infer_paper_ids_from_text(
    text: str,
    papers: list[dict[str, Any]],
    profile_order_map: dict[int, str] | None = None,
    max_matches: int = 1,
) -> list[str]:
    """从候选方向自然语言描述中保守推断目标 paper_id。"""
    profile_order_map = profile_order_map or {}
    number_match = re.search(r"论文\s*(\d+)", text)
    if number_match:
        paper_id = profile_order_map.get(int(number_match.group(1)))
        if paper_id:
            return [paper_id]

    normalized_text = normalize_for_match(text)
    scored: list[tuple[int, str]] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        title = str(paper.get("title") or "")
        if paper_id and paper_id.lower() in text.lower():
            return [paper_id]

        tokens = title_tokens(title)
        if not tokens:
            continue
        score = sum(1 for token in tokens if token in normalized_text)

        title_normalized = normalize_for_match(title)
        if title_normalized and title_normalized in normalized_text:
            score += 10

        if score > 0:
            scored.append((score, paper_id))

    scored.sort(reverse=True)
    return [paper_id for _, paper_id in scored[:max_matches] if paper_id]


def build_where(section_keys: list[str] | None = None, paper_ids: list[str] | None = None) -> dict[str, Any] | None:
    clauses = []
    if section_keys:
        clauses.append({"section_key": {"$in": section_keys}})
    if paper_ids:
        clauses.append({"paper_id": {"$in": paper_ids}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_claim_bucket_queries(
    candidate: dict[str, Any],
    papers: list[dict[str, Any]] | None = None,
    profile_order_map: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """把候选方向拆成 base/module/connection 三类查证 query。"""
    source_basis = candidate.get("source_basis") or {}
    papers = papers or []
    base_paper_ids = infer_paper_ids_from_text(
        str(source_basis.get("base_paper") or ""),
        papers,
        profile_order_map=profile_order_map,
    )
    module_paper_ids = infer_paper_ids_from_text(
        str(source_basis.get("module_paper") or ""),
        papers,
        profile_order_map=profile_order_map,
    )
    connection_paper_ids = list(dict.fromkeys(base_paper_ids + module_paper_ids))
    common_lines = [
        f"candidate_id: {candidate.get('candidate_id')}",
        f"candidate_direction: {candidate.get('candidate_direction')}",
        f"research_hypothesis: {candidate.get('research_hypothesis')}",
    ]

    return [
        {
            "bucket": "base_problem",
            "section_priorities": BUCKET_SECTION_PRIORITIES["base_problem"],
            "target_paper_ids": base_paper_ids,
            "query": "\n".join(
                common_lines
                + [
                    f"base_paper: {source_basis.get('base_paper')}",
                    f"combination_reason: {source_basis.get('combination_reason')}",
                    "查证重点：base 论文是否真的存在候选方向所说的问题、局限、遗留空间或可改进点。",
                ]
            ),
        },
        {
            "bucket": "module_method",
            "section_priorities": BUCKET_SECTION_PRIORITIES["module_method"],
            "target_paper_ids": module_paper_ids,
            "query": "\n".join(
                common_lines
                + [
                    f"module_paper: {source_basis.get('module_paper')}",
                    f"combination_reason: {source_basis.get('combination_reason')}",
                    "查证重点：module 论文是否真的提供了候选方向所说的可迁移方法模块，其输入、输出和核心机制是什么。",
                ]
            ),
        },
        {
            "bucket": "connection",
            "section_priorities": BUCKET_SECTION_PRIORITIES["connection"],
            "target_paper_ids": connection_paper_ids,
            "query": "\n".join(
                common_lines
                + [
                    f"base_paper: {source_basis.get('base_paper')}",
                    f"module_paper: {source_basis.get('module_paper')}",
                    f"compatibility_assumption: {source_basis.get('compatibility_assumption')}",
                    f"questions_to_verify: {json.dumps(candidate.get('questions_to_verify') or [], ensure_ascii=False)}",
                    "查证重点：module 的输入、输出、训练机制和依赖条件是否能接到 base 问题，是否存在任务设定或优化目标冲突。",
                ]
            ),
        },
    ]


def result_key(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return str(result.get("content") or "")[:300]


def annotate_results(results: list[dict[str, Any]], bucket: str, stage: str) -> list[dict[str, Any]]:
    annotated = []
    for result in results:
        item = dict(result)
        metadata = dict(item.get("metadata") or {})
        metadata["retrieval_bucket"] = bucket
        metadata["retrieval_stage"] = stage
        item["metadata"] = metadata
        item["retrieval_bucket"] = bucket
        item["retrieval_stage"] = stage
        annotated.append(item)
    return annotated


async def query_one_paper_with_section_priority(
    db_id: str,
    bucket: str,
    query_text: str,
    section_priorities: list[str],
    paper_id: str,
    top_k: int,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """对单篇目标论文执行章节优先召回。"""
    section_results = await knowledge_base.aquery(
        query_text,
        db_id=db_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        where=build_where(section_priorities, [paper_id]),
    )
    collected = annotate_results(section_results, bucket, "target_paper_section_priority")
    seen = {result_key(result) for result in collected}

    paper_fallback_added_count = 0
    global_fallback_added_count = 0
    if len(collected) < top_k:
        paper_results = await knowledge_base.aquery(
            query_text,
            db_id=db_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            where=build_where(paper_ids=[paper_id]),
        )
        for result in annotate_results(paper_results, bucket, "target_paper_fulltext_fallback"):
            key = result_key(result)
            if key in seen:
                continue
            seen.add(key)
            collected.append(result)
            paper_fallback_added_count += 1
            if len(collected) >= top_k:
                break

    if len(collected) < top_k:
        full_results = await knowledge_base.aquery(
            query_text,
            db_id=db_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        for result in annotate_results(full_results, bucket, "global_fulltext_fallback"):
            key = result_key(result)
            if key in seen:
                continue
            seen.add(key)
            collected.append(result)
            global_fallback_added_count += 1
            if len(collected) >= top_k:
                break

    return collected[:top_k], len(section_results), paper_fallback_added_count, global_fallback_added_count


async def query_bucket_with_section_priority(
    db_id: str,
    bucket: str,
    query_text: str,
    section_priorities: list[str],
    target_paper_ids: list[str],
    top_k: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    """每桶先查目标论文优先章节，不足再回退。connection 桶按目标论文均衡取证。"""
    if bucket == "connection" and len(target_paper_ids) > 1:
        per_paper_quota = max(1, top_k // len(target_paper_ids))
        remainder = max(0, top_k - per_paper_quota * len(target_paper_ids))
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        section_result_count = 0
        paper_fallback_added_count = 0
        global_fallback_added_count = 0
        balanced_allocation: dict[str, int] = {}

        for index, paper_id in enumerate(target_paper_ids):
            paper_top_k = per_paper_quota + (1 if index < remainder else 0)
            balanced_allocation[paper_id] = paper_top_k
            paper_results, section_count, paper_fallback_count, global_fallback_count = (
                await query_one_paper_with_section_priority(
                    db_id=db_id,
                    bucket=bucket,
                    query_text=query_text,
                    section_priorities=section_priorities,
                    paper_id=paper_id,
                    top_k=paper_top_k,
                    similarity_threshold=similarity_threshold,
                )
            )
            section_result_count += section_count
            paper_fallback_added_count += paper_fallback_count
            global_fallback_added_count += global_fallback_count
            for result in paper_results:
                key = result_key(result)
                if key in seen:
                    continue
                seen.add(key)
                collected.append(result)

        return {
            "bucket": bucket,
            "bucket_label": BUCKET_LABELS[bucket],
            "query": query_text,
            "section_priorities": section_priorities,
            "target_paper_ids": target_paper_ids,
            "balanced_allocation": balanced_allocation,
            "section_result_count": section_result_count,
            "paper_fallback_added_count": paper_fallback_added_count,
            "global_fallback_added_count": global_fallback_added_count,
            "results": collected[:top_k],
        }

    section_results = await knowledge_base.aquery(
        query_text,
        db_id=db_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        where=build_where(section_priorities, target_paper_ids),
    )
    collected = annotate_results(section_results, bucket, "target_paper_section_priority")
    seen = {result_key(result) for result in collected}

    paper_fallback_results: list[dict[str, Any]] = []
    global_fallback_results: list[dict[str, Any]] = []
    if len(collected) < top_k:
        paper_results = []
        if target_paper_ids:
            paper_results = await knowledge_base.aquery(
                query_text,
                db_id=db_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                where=build_where(paper_ids=target_paper_ids),
            )
        for result in annotate_results(paper_results, bucket, "target_paper_fulltext_fallback"):
            key = result_key(result)
            if key in seen:
                continue
            seen.add(key)
            paper_fallback_results.append(result)
            collected.append(result)
            if len(collected) >= top_k:
                break

    if len(collected) < top_k:
        full_results = await knowledge_base.aquery(
            query_text,
            db_id=db_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        for result in annotate_results(full_results, bucket, "global_fulltext_fallback"):
            key = result_key(result)
            if key in seen:
                continue
            seen.add(key)
            global_fallback_results.append(result)
            collected.append(result)
            if len(collected) >= top_k:
                break

    return {
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS[bucket],
        "query": query_text,
        "section_priorities": section_priorities,
        "target_paper_ids": target_paper_ids,
        "balanced_allocation": {},
        "section_result_count": len(section_results),
        "paper_fallback_added_count": len(paper_fallback_results),
        "global_fallback_added_count": len(global_fallback_results),
        "results": collected[:top_k],
    }


async def retrieve_candidate_evidence(
    db_id: str,
    candidate: dict[str, Any],
    papers: list[dict[str, Any]] | None,
    profile_order_map: dict[int, str] | None,
    top_k_per_bucket: int = 4,
    similarity_threshold: float = 0.0,
) -> dict[str, Any]:
    """按三桶召回候选方向证据。"""
    bucket_queries = build_claim_bucket_queries(candidate, papers, profile_order_map)
    bucket_results: dict[str, Any] = {}
    merged_results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for bucket_query in bucket_queries:
        bucket_result = await query_bucket_with_section_priority(
            db_id=db_id,
            bucket=bucket_query["bucket"],
            query_text=bucket_query["query"],
            section_priorities=bucket_query["section_priorities"],
            target_paper_ids=bucket_query["target_paper_ids"],
            top_k=top_k_per_bucket,
            similarity_threshold=similarity_threshold,
        )
        bucket_results[bucket_query["bucket"]] = bucket_result
        for result in bucket_result["results"]:
            key = result_key(result)
            if key in seen:
                continue
            seen.add(key)
            merged_results.append(result)

    return {
        "query_mode": "claim_buckets",
        "candidate": candidate,
        "bucket_queries": bucket_queries,
        "bucket_results": bucket_results,
        "merged_results": merged_results,
    }


def format_result_block(index: int, result: dict[str, Any], max_chars_per_result: int) -> list[str]:
    metadata = result.get("metadata") or {}
    content = str(result.get("content") or "").strip()
    return [
        f"### RAG 证据 {index}",
        "",
        f"- score：{result.get('score')}",
        f"- paper_id：{metadata.get('paper_id', '')}",
        f"- title：{metadata.get('title', '')}",
        f"- section：{metadata.get('section_key', '')} | {metadata.get('section_title', '')}",
        f"- pages：{metadata.get('page_start', '')}-{metadata.get('page_end', '')}",
        f"- chunk_id：{metadata.get('chunk_id', '')}",
        f"- retrieval_bucket：{metadata.get('retrieval_bucket', '')}",
        f"- retrieval_stage：{metadata.get('retrieval_stage', '')}",
        "",
        "```text",
        content[:max_chars_per_result],
        "```",
        "",
    ]


def format_evidence_markdown(query_text: str, results: list[dict[str, Any]], max_chars_per_result: int) -> str:
    lines = [
        "# PDF RAG 召回证据",
        "",
        "说明：以下片段来自已下载 PDF 的 Chroma 向量召回，可作为证据查证 Agent 的 extra_evidence 输入。",
        "注意：score 只用于同一次查询内的排序参考，不作为绝对可信分数。",
        "",
        "## 查询文本",
        "",
        "```text",
        query_text[:3000],
        "```",
        "",
        "## 召回片段",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(format_result_block(index, result, max_chars_per_result))
    return "\n".join(lines)


def format_bucket_evidence_markdown(bucket_output: dict[str, Any], max_chars_per_result: int) -> str:
    lines = [
        "# PDF RAG 召回证据",
        "",
        "说明：以下片段来自已下载 PDF 的 Chroma 向量召回，可作为证据查证 Agent 的 extra_evidence 输入。",
        "注意：score 只用于同一次查询内的排序参考，不作为绝对可信分数。",
        "召回策略：按 base_problem / module_method / connection 三桶检索；每桶先查优先章节，不足再全文补齐。",
        "",
    ]

    evidence_index = 1
    for bucket in ["base_problem", "module_method", "connection"]:
        bucket_result = bucket_output["bucket_results"][bucket]
        lines.extend(
            [
                f"## {bucket}：{bucket_result['bucket_label']}",
                "",
                f"- section_priorities：{', '.join(bucket_result['section_priorities'])}",
                f"- target_paper_ids：{', '.join(bucket_result['target_paper_ids'])}",
                f"- balanced_allocation：{json.dumps(bucket_result.get('balanced_allocation') or {}, ensure_ascii=False)}",
                f"- section_result_count：{bucket_result['section_result_count']}",
                f"- paper_fallback_added_count：{bucket_result['paper_fallback_added_count']}",
                f"- global_fallback_added_count：{bucket_result['global_fallback_added_count']}",
                "",
                "### 本桶查询文本",
                "",
                "```text",
                bucket_result["query"][:3000],
                "```",
                "",
            ]
        )
        for result in bucket_result["results"]:
            lines.extend(format_result_block(evidence_index, result, max_chars_per_result))
            evidence_index += 1
    return "\n".join(lines)
