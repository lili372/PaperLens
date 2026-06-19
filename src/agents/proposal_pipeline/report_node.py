"""选题建议系统 - 最终报告节点。

职责：不调用大模型，直接从最终候选、查证和审查产物中拼装中文报告。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PAPER_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}v\d+")
RAG_REF_PATTERN = re.compile(r"RAG\s*证据\s*(\d+)")

STATUS_LABELS = {
    "supported": "支持",
    "partially_supported": "部分支持",
    "unsupported": "未支持",
    "contradicted": "存在反证",
    "keep": "保留",
    "revise_keep": "修改后保留",
    "defer": "暂缓",
    "reject": "拒绝",
    "pass": "通过",
    "warning": "警告",
    "fail": "失败",
    "strong": "强",
    "medium": "中",
    "weak": "弱",
    "high": "高",
    "low": "低",
}


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    """只接受真实列表，其他值视为空列表。"""
    return value if isinstance(value, list) else []


def find_summary_item(summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """从批量审查汇总中找到某个候选方向。"""
    for item in as_list(summary.get("items")):
        if item.get("candidate_id") == candidate_id:
            return item
    return {}


def read_optional_json(path_text: str | None) -> dict[str, Any]:
    """读取可选 JSON 文件，缺失时返回空对象。"""
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def first_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """取前 N 条，保持原顺序。"""
    return items[:limit] if len(items) > limit else items


def zh_status(status: Any) -> str:
    """把内部枚举转成中文展示。"""
    status_text = str(status or "")
    return STATUS_LABELS.get(status_text, status_text)


def display_text(value: Any) -> str:
    """把少量内部术语替换成报告里的中文表达。"""
    text = str(value or "")
    replacements = {
        "base_problem_status": "基础问题状态",
        "module_status": "方法模块状态",
        "connection_status": "组合连接状态",
        "partially_supported": "部分支持",
        "supported": "支持",
        "unsupported": "未支持",
        "contradicted": "存在反证",
        "defer": "暂缓",
        "reject": "拒绝",
        "base问题": "基础问题",
        "base论文": "基础论文",
        "module是": "方法模块是",
        "module（": "方法模块（",
        "module论文": "方法模块论文",
        "module方法": "方法模块",
        "作为module": "作为方法模块",
        "connection": "组合连接",
        "RAG证据": "PDF 证据",
        "RAG 证据": "PDF 证据",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def compact_candidate_title(direction: Any, max_chars: int = 24) -> str:
    """从完整候选方向派生前端展示用短标题。"""
    text = re.sub(r"\s+", "", str(direction or ""))
    if not text:
        return ""

    pattern_titles = [
        (("ProPML", "POMDP", "BCE"), "ProPML替换BCE消歧损失"),
        (("POMDP", "高秩"), "POMDP高秩约束消歧"),
        (("POMDP", "迭代传播"), "POMDP迭代传播伪标签"),
        (("Schirn", "跨模态"), "Schirn连续置信度输入"),
        (("Schirn", "双级图正则"), "Schirn双级图正则"),
        (("高秩", "稀疏", "连续"), "连续概率高秩稀疏消歧"),
        (("标签相关性", "实例平滑"), "标签相关性与实例平滑"),
    ]
    for keywords, title in pattern_titles:
        if all(keyword in text for keyword in keywords):
            return title[:max_chars]

    return ""


def unique_keep_order(values: list[str]) -> list[str]:
    """按首次出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_paper_ids(text: Any) -> list[str]:
    """从任意文本中抽取 arXiv paper_id。"""
    return PAPER_ID_PATTERN.findall(str(text or ""))


def build_profile_order_map_from_rag(rag_query: dict[str, Any]) -> dict[str, str]:
    """从 RAG 查询记录反推候选里“论文 N”对应的真实论文 ID。"""
    mapping: dict[str, str] = {}
    candidate = rag_query.get("candidate") or {}
    source_basis = candidate.get("source_basis") or {}
    for key in ("base_paper", "module_paper"):
        text = str(source_basis.get(key, ""))
        match = re.search(r"论文\s*(\d+)", text)
        bucket = "base_problem" if key == "base_paper" else "module_method"
        bucket_queries = [
            query for query in as_list(rag_query.get("bucket_queries"))
            if query.get("bucket") == bucket
        ]
        target_ids = as_list(bucket_queries[0].get("target_paper_ids")) if bucket_queries else []
        if match and target_ids:
            mapping[f"paper_profile_{match.group(1)}"] = str(target_ids[0])
            mapping[f"论文{match.group(1)}"] = str(target_ids[0])
    return mapping


def build_profile_order_map_from_topic(topic_debug_path: Path) -> dict[str, str]:
    """从候选生成调试文件的画像输入顺序反推“论文画像 N”对应的论文 ID。"""
    if not topic_debug_path.exists():
        return {}

    mapping: dict[str, str] = {}
    profile_index = 1
    for line in topic_debug_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("- ") or "paper_profile_" not in text:
            continue
        match = re.search(r"paper_profile_(.+?)\.md", text)
        if not match:
            continue
        paper_id = match.group(1)
        mapping[f"paper_profile_{profile_index}"] = paper_id
        mapping[f"论文画像{profile_index}"] = paper_id
        mapping[f"论文画像 {profile_index}"] = paper_id
        mapping[f"论文{profile_index}"] = paper_id
        profile_index += 1
    return mapping


def build_rag_ref_map(rag_query: dict[str, Any]) -> dict[str, str]:
    """把 RAG 证据序号映射到真实论文 ID。"""
    mapping: dict[str, str] = {}
    index = 1
    bucket_results = rag_query.get("bucket_results") or {}
    for bucket in ("base_problem", "module_method", "connection"):
        results = as_list((bucket_results.get(bucket) or {}).get("results"))
        for result in results:
            metadata = result.get("metadata") or {}
            paper_id = metadata.get("paper_id") or metadata.get("file_id")
            if paper_id:
                mapping[f"RAG 证据 {index}"] = str(paper_id)
                mapping[f"RAG证据{index}"] = str(paper_id)
            index += 1
    return mapping


def resolve_source_ids(source: Any, source_map: dict[str, str]) -> list[str]:
    """把证据来源文本解析为论文 ID 列表。"""
    text = str(source or "")
    ids = extract_paper_ids(text)
    normalized = re.sub(r"\s+", " ", text)
    for source_key, paper_id in source_map.items():
        if source_key in text or source_key in normalized:
            ids.append(paper_id)
    for match in RAG_REF_PATTERN.findall(text):
        paper_id = source_map.get(f"RAG 证据 {match}") or source_map.get(f"RAG证据{match}")
        if paper_id:
            ids.append(paper_id)
    return unique_keep_order(ids)


def format_source_strength(source_ids: list[str], strength: Any) -> str:
    """格式化证据来源和强度。"""
    source_text = "、".join(source_ids) if source_ids else "未定位到具体论文 ID"
    strength_text = zh_status(strength)
    return f"来源：{source_text}；证据强度：{strength_text or '未标注'}"


def build_scope(
    search_result: dict[str, Any],
    pdf_manifest: dict[str, Any],
    final_package: dict[str, Any],
) -> dict[str, Any]:
    """生成报告范围摘要。"""
    search_plan = search_result.get("search_plan") or pdf_manifest.get("search_plan") or {}
    return {
        "user_request": search_result.get("user_request") or pdf_manifest.get("user_request") or "",
        "base_query": search_plan.get("base_query", ""),
        "module_terms": search_plan.get("module_terms") or [],
        "start_date": search_plan.get("start_date"),
        "search_paper_count": (
            search_result.get("raw_count")
            or search_result.get("count")
            or len(as_list(search_result.get("papers")))
        ),
        "pdf_success_count": pdf_manifest.get("success_count"),
        "pdf_failed_count": pdf_manifest.get("failed_count"),
        "final_candidate_count": final_package.get("final_candidate_count", 0),
        "deferred_candidate_count": final_package.get("deferred_candidate_count", 0),
        "rejected_candidate_count": final_package.get("rejected_candidate_count", 0),
    }


def build_candidate_report_item(
    final_item: dict[str, Any],
    summary_item: dict[str, Any],
    topic_profile_map: dict[str, str],
) -> dict[str, Any]:
    """把单个最终候选方向整理成报告结构。"""
    candidate_id = str(final_item.get("candidate_id", ""))
    candidate = final_item.get("final_candidate") or {}
    retrieval = read_optional_json((final_item.get("paths") or {}).get("retrieval"))
    review = read_optional_json((final_item.get("paths") or {}).get("review"))
    rag_query_path = (summary_item.get("paths") or {}).get("rag_query")
    rag_query = read_optional_json(rag_query_path)
    source_map = {
        **topic_profile_map,
        **build_profile_order_map_from_rag(rag_query),
        **build_rag_ref_map(rag_query),
    }
    claim_checks = retrieval.get("claim_checks") or summary_item.get("claim_checks") or {}
    supporting = first_items(as_list(retrieval.get("supporting_evidence")), 3)
    risks = first_items(as_list(retrieval.get("risk_evidence")), 3)
    target_paper_ids = unique_keep_order(
        [
            str(paper_id)
            for query in as_list(rag_query.get("bucket_queries"))
            for paper_id in as_list(query.get("target_paper_ids"))
        ]
    )
    if not target_paper_ids:
        source_refs: list[Any] = []
        for claim in claim_checks.values():
            if isinstance(claim, dict):
                source_refs.extend(as_list(claim.get("evidence_refs")))
        for text in (candidate.get("source_basis") or {}).values():
            source_refs.append(text)
        target_paper_ids = unique_keep_order(
            [
                paper_id
                for ref in source_refs
                for paper_id in resolve_source_ids(ref, source_map)
            ]
        )

    return {
        "candidate_id": candidate_id,
        "candidate_title": candidate.get("candidate_title") or compact_candidate_title(candidate.get("candidate_direction", "")),
        "candidate_direction": candidate.get("candidate_direction", ""),
        "review_decision": final_item.get("review_decision"),
        "guard_result": final_item.get("guard_result"),
        "target_paper_ids": target_paper_ids,
        "research_value": candidate.get("research_value", ""),
        "why_worth_reading": (
            review.get("decision_reason")
            or summary_item.get("final_advice")
            or final_item.get("final_advice", "")
        ),
        "claim_checks": {
            "base_problem": claim_checks.get("base_problem") or {},
            "module_method": claim_checks.get("module_method") or {},
            "connection": claim_checks.get("connection") or {},
        },
        "source_map": source_map,
        "supporting_evidence": supporting,
        "risk_evidence": risks,
        "validation_path": review.get("validation_path") or "",
        "questions_to_verify": as_list(candidate.get("questions_to_verify")),
        "final_advice": review.get("final_advice") or final_item.get("final_advice", ""),
        "paths": final_item.get("paths") or {},
    }


def build_deferred_report_item(
    deferred_item: dict[str, Any],
    summary_item: dict[str, Any],
) -> dict[str, Any]:
    """把暂缓候选方向整理成报告结构。"""
    candidate_id = str(deferred_item.get("candidate_id", ""))
    candidate = deferred_item.get("candidate") or {}
    retrieval = read_optional_json((deferred_item.get("paths") or {}).get("retrieval"))
    review = read_optional_json((deferred_item.get("paths") or {}).get("review"))
    risks = first_items(as_list(retrieval.get("risk_evidence")), 3)

    return {
        "candidate_id": candidate_id,
        "candidate_title": candidate.get("candidate_title") or compact_candidate_title(candidate.get("candidate_direction", "")),
        "candidate_direction": candidate.get("candidate_direction", ""),
        "defer_reason": review.get("final_advice") or deferred_item.get("final_advice", ""),
        "review_features": review.get("review_features") or deferred_item.get("review_features") or {},
        "risk_evidence": risks,
        "evidence_gaps": as_list(retrieval.get("evidence_gaps")) or as_list(summary_item.get("evidence_gaps")),
        "paths": deferred_item.get("paths") or {},
    }


def candidate_priority_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    """候选方向排序：证据越稳、先读论文越明确、风险越少越靠前。"""
    claim_checks = item.get("claim_checks") or {}
    supported_count = sum(
        1
        for claim in claim_checks.values()
        if isinstance(claim, dict) and claim.get("status") == "supported"
    )
    target_count = len(as_list(item.get("target_paper_ids")))
    risk_count = len(as_list(item.get("risk_evidence")))
    return (-supported_count, -target_count, risk_count, str(item.get("candidate_id") or ""))


def split_display_candidates(final_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拆分主推荐和其他可考虑方向。"""
    sorted_items = sorted(final_items, key=candidate_priority_key)
    return sorted_items[:3], sorted_items[3:]


def bullet(text: str) -> str:
    """生成 Markdown 列表行。"""
    return f"- {text}" if text else "- 暂无"


def format_claim_line(label: str, claim: dict[str, Any], source_map: dict[str, str]) -> str:
    """格式化一个 claim_checks 行。"""
    status = zh_status(claim.get("status", ""))
    note = display_text(claim.get("note", ""))
    source_ids = []
    for ref in as_list(claim.get("evidence_refs")):
        source_ids.extend(resolve_source_ids(ref, source_map))
    source_text = "、".join(unique_keep_order(source_ids))
    source_suffix = f"；来源：{source_text}" if source_text else ""
    return f"- {label}：{status}。{note}{source_suffix}"


def format_evidence_lines(
    evidence_items: list[dict[str, Any]],
    source_map: dict[str, str],
    fallback_paper_ids: list[str],
) -> list[str]:
    """格式化支持材料或风险材料。"""
    if not evidence_items:
        return ["- 暂无结构化证据。"]
    lines: list[str] = []
    for evidence in evidence_items:
        claim = display_text(evidence.get("supported_claim") or evidence.get("risk_point") or "证据")
        summary = display_text(evidence.get("evidence_summary", ""))
        source_ids = resolve_source_ids(evidence.get("evidence_source", ""), source_map)
        if not source_ids:
            source_ids = fallback_paper_ids
        strength = evidence.get("evidence_strength") or evidence.get("risk_strength") or ""
        lines.append(f"- {claim}：{summary}（{format_source_strength(source_ids, strength)}）")
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    """把报告结构渲染成中文 Markdown。"""
    scope = report["scope"]
    lines = [
        "# 偏多标记学习候选研究方向报告",
        "",
        "## 本次分析范围",
        "",
        f"- 检索主题：{scope.get('base_query')}",
        f"- 模块词：{'、'.join(scope.get('module_terms') or [])}",
        f"- 时间范围：{scope.get('start_date') or '未限制'} 之后",
        f"- arXiv 去重论文数：{scope.get('search_paper_count')}",
        f"- PDF 成功解析基础：成功 {scope.get('pdf_success_count')} 篇，失败 {scope.get('pdf_failed_count')} 篇",
        f"- 最终推荐方向：{scope.get('final_candidate_count')} 个；暂缓方向：{scope.get('deferred_candidate_count')} 个",
        "",
        "## 推荐继续精读 / 小实验的方向",
        "",
    ]

    if not (report.get("primary_candidates") or report.get("final_candidates")):
        lines.extend(
            [
                report.get("empty_recommendation_message") or "当前材料没有形成可靠推荐方向。",
                "",
            ]
        )

    for item in report.get("primary_candidates") or report["final_candidates"]:
        claims = item["claim_checks"]
        source_map = item.get("source_map") or {}
        target_paper_ids = item.get("target_paper_ids") or []
        lines.extend(
            [
                f"### {item['candidate_id']}：{item.get('candidate_title') or item['candidate_direction']}",
                "",
                f"- 建议先读：{'、'.join(item.get('target_paper_ids') or []) or '暂无'}",
                bullet(f"方向说明：{display_text(item.get('candidate_direction', ''))}"),
                bullet(f"为什么值得看：{display_text(item.get('why_worth_reading', ''))}"),
                bullet(f"研究价值：{display_text(item.get('research_value', ''))}"),
                "",
                "证据支持：",
                format_claim_line("基础问题", claims.get("base_problem") or {}, source_map),
                format_claim_line("方法模块", claims.get("module_method") or {}, source_map),
                format_claim_line("组合连接", claims.get("connection") or {}, source_map),
                "",
                "关键支持材料：",
                *format_evidence_lines(item["supporting_evidence"], source_map, target_paper_ids),
                "",
                "主要风险：",
                *format_evidence_lines(item["risk_evidence"], source_map, target_paper_ids),
                "",
                bullet(f"建议先做的验证：{display_text(item.get('validation_path') or item.get('final_advice', ''))}"),
                "",
            ]
        )

    if report.get("secondary_candidates"):
        lines.extend(["## 其他可考虑方向", ""])
        for item in report["secondary_candidates"]:
            lines.append(
                f"- {item['candidate_id']}：{item.get('candidate_title') or item['candidate_direction']}。"
                f"方向说明：{display_text(item.get('candidate_direction', ''))}"
            )
        lines.append("")

    if report["deferred_candidates"]:
        lines.extend(["## 未进入主推荐的方向", ""])
    for item in report["deferred_candidates"]:
        lines.append(
            f"- {item['candidate_id']}：{item.get('candidate_title') or item['candidate_direction']}。"
            f"方向说明：{display_text(item.get('candidate_direction', ''))}。"
            f"暂缓原因：{display_text(item.get('defer_reason', ''))}"
        )
    lines.append("")

    lines.extend(
        [
            "## 使用说明",
            "",
            "这份报告用于决定下一步精读和小实验优先级，不证明任何方向一定成立。",
            "报告只拼装前序查证、审查和单次修改结果，不新增判断，也没有调用大模型做二次总结。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """从产物路径构建最终报告对象。"""
    search_result = read_json(Path(args.search_results))
    search_results_all_path = Path(args.search_results).with_name("search_results_all.json")
    if search_results_all_path.exists():
        search_result_all = read_json(search_results_all_path)
        if isinstance(search_result_all, dict):
            search_result = {**search_result, "raw_count": search_result_all.get("raw_count")}
    pdf_manifest = read_json(Path(args.pdf_manifest))
    final_package = read_json(Path(args.final_candidates))
    deferred_candidates = read_json(Path(args.deferred_candidates))
    summary = read_json(Path(args.review_summary))
    topic_profile_map = build_profile_order_map_from_topic(
        Path(args.final_candidates).parents[1] / "topic_proposals_probe_json.md"
    )

    if not isinstance(final_package, dict):
        raise ValueError("final_candidates 文件必须是对象")
    if not isinstance(deferred_candidates, list):
        raise ValueError("deferred_candidates 文件必须是数组")
    if not isinstance(summary, dict):
        raise ValueError("review_summary 文件必须是对象")

    final_items = []
    for final_item in as_list(final_package.get("final_candidates")):
        candidate_id = str(final_item.get("candidate_id", ""))
        final_items.append(
            build_candidate_report_item(final_item, find_summary_item(summary, candidate_id), topic_profile_map)
        )
    primary_items, secondary_items = split_display_candidates(final_items)

    deferred_items = []
    if len(final_items) < 3:
        for deferred_item in deferred_candidates:
            candidate_id = str(deferred_item.get("candidate_id", ""))
            deferred_items.append(build_deferred_report_item(deferred_item, find_summary_item(summary, candidate_id)))

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "proposal_report_node",
        "llm_used": False,
        "scope": build_scope(search_result, pdf_manifest, final_package),
        "empty_recommendation_message": (
            "当前材料没有形成可靠推荐方向。建议扩大检索范围、调整模块词，或补充更多可分析论文后重试。"
            if not final_items
            else ""
        ),
        "final_candidates": final_items,
        "primary_candidates": primary_items,
        "secondary_candidates": secondary_items,
        "deferred_candidates": deferred_items,
        "source_files": {
            "search_results": str(Path(args.search_results)),
            "pdf_manifest": str(Path(args.pdf_manifest)),
            "final_candidates": str(Path(args.final_candidates)),
            "deferred_candidates": str(Path(args.deferred_candidates)),
            "review_summary": str(Path(args.review_summary)),
        },
    }


def write_report(args: argparse.Namespace) -> dict[str, str]:
    """生成 Markdown 和 JSON 报告文件。"""
    report = build_report(args)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(report), encoding="utf-8")
    write_json(out_json, report)
    return {"report_md": str(out_md), "report_json": str(out_json)}
