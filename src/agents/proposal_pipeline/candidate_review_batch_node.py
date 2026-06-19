"""选题建议系统 - 候选方向批量查证与审查节点。

用途：对候选方向批量执行 RAG 召回、证据查证和可做性审查。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.agents.proposal_pipeline.pdf_parse import available_pdf_papers
from src.agents.proposal_pipeline.rag_retrieval import (
    build_profile_order_map,
    format_bucket_evidence_markdown,
    retrieve_candidate_evidence,
)
from src.agents.proposal_pipeline.retrieval_node import run_retrieval
from src.agents.proposal_pipeline.review_node import run_review
from src.core.config import config

STATUS_VALUES = {"supported", "partially_supported", "unsupported", "contradicted"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidates(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("候选方向文件必须是 JSON 数组")
    return data


def select_candidates(candidates: list[dict[str, Any]], candidate_ids: list[str] | None) -> list[dict[str, Any]]:
    if not candidate_ids:
        return candidates
    wanted = set(candidate_ids)
    selected = [candidate for candidate in candidates if candidate.get("candidate_id") in wanted]
    missing = wanted - {str(candidate.get("candidate_id")) for candidate in selected}
    if missing:
        raise ValueError(f"未找到候选方向: {sorted(missing)}")
    return selected


def fallback_review_item(
    candidate: dict[str, Any],
    reason: str,
    paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """把单候选异常降级为暂缓，并保留错误信息。"""
    candidate_id = str(candidate.get("candidate_id"))
    return {
        "candidate_id": candidate_id,
        "candidate_direction": candidate.get("candidate_direction"),
        "rag_result_count": 0,
        "verification_conclusion": "insufficient",
        "claim_checks": {},
        "review_decision": "defer",
        "review_features": {
            "base_problem_status": "unsupported",
            "module_status": "unsupported",
            "connection_status": "unsupported",
            "dependency_conflict": False,
            "critical_evidence_gap_count": 1,
            "major_risk_count": 0,
            "unsupported_assumption_count": 0,
        },
        "required_revision": "",
        "final_advice": f"该候选方向处理失败，已暂缓：{reason}",
        "business_fallback": {
            "type": "candidate_processing_error",
            "reason": reason,
        },
        "paths": paths or {},
    }


def normalize_review_consistency(item: dict[str, Any]) -> dict[str, Any]:
    """按硬规则修正查证/审查不一致，宁可降级也不让强结论通过。"""
    features = item.get("review_features") or {}
    decision = item.get("review_decision")
    original_decision = decision
    reason = ""

    base = features.get("base_problem_status")
    module = features.get("module_status")
    connection = features.get("connection_status")
    dependency_conflict = features.get("dependency_conflict") is True
    statuses = [base, module, connection]

    if decision == "keep" and statuses != ["supported", "supported", "supported"]:
        decision = "revise_keep"
        reason = "keep 需要 base/module/connection 全部 supported，已降级为 revise_keep。"
    if decision == "revise_keep" and (
        dependency_conflict or any(status == "contradicted" for status in statuses)
    ):
        decision = "defer"
        reason = "revise_keep 不能存在 contradicted 或 dependency_conflict=true，已降级为 defer。"
    if decision == "keep" and dependency_conflict:
        decision = "defer"
        reason = "keep 不能存在 dependency_conflict=true，已降级为 defer。"
    if decision not in {"keep", "revise_keep", "defer", "reject"}:
        decision = "defer"
        reason = f"review_decision 不合法：{original_decision}，已降级为 defer。"

    if decision != original_decision:
        item["original_review_decision"] = original_decision
        item["review_decision"] = decision
        item["business_fallback"] = {
            "type": "review_consistency_downgrade",
            "reason": reason,
        }
        item["final_advice"] = f"{item.get('final_advice') or ''}（系统兜底：{reason}）".strip()
    return item


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 候选方向批量查证 / 审查汇总",
        "",
        f"- db_id：{summary['db_id']}",
        f"- candidate_count：{summary['candidate_count']}",
        "",
        "| candidate_id | verification | review | base | module | connection | 建议 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in summary["items"]:
        features = item.get("review_features") or {}
        lines.append(
            "| {candidate_id} | {verification_conclusion} | {review_decision} | {base} | {module} | {connection} | {advice} |".format(
                candidate_id=item.get("candidate_id", ""),
                verification_conclusion=item.get("verification_conclusion", ""),
                review_decision=item.get("review_decision", ""),
                base=features.get("base_problem_status", ""),
                module=features.get("module_status", ""),
                connection=features.get("connection_status", ""),
                advice=str(item.get("final_advice", "")).replace("|", "/"),
            )
        )
    return "\n".join(lines)


async def run_batch(args: argparse.Namespace) -> None:
    db_id = args.existing_db_id
    config.set("tmp_db_id", db_id)

    candidates_path = Path(args.candidates)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    profile_paths = [Path(path) for path in args.profiles]

    candidates = select_candidates(load_candidates(candidates_path), args.candidate_ids)
    manifest = read_json(manifest_path)
    papers = available_pdf_papers(manifest)
    profile_order_map = build_profile_order_map(profile_paths)

    summary_items = []
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency = max(1, config.get_int("proposal_review_concurrency", 3))
    semaphore = asyncio.Semaphore(concurrency)

    async def process_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id"))
        prefix = out_dir / candidate_id
        rag_query_path = prefix.with_name(f"{candidate_id}_rag_query.json")
        evidence_path = prefix.with_name(f"{candidate_id}_extra_evidence.md")
        retrieval_path = prefix.with_name(f"{candidate_id}_retrieval.json")
        review_path = prefix.with_name(f"{candidate_id}_review.json")
        error_path = prefix.with_name(f"{candidate_id}_error.json")

        paths = {
            "rag_query": str(rag_query_path),
            "extra_evidence": str(evidence_path),
            "retrieval": str(retrieval_path),
            "review": str(review_path),
            "error": str(error_path),
        }
        try:
            bucket_output = await retrieve_candidate_evidence(
                db_id=db_id,
                candidate=candidate,
                papers=papers,
                profile_order_map=profile_order_map,
                top_k_per_bucket=args.top_k_per_bucket,
                similarity_threshold=args.similarity_threshold,
            )
            rag_query_output = {
                "db_id": db_id,
                "query_mode": "claim_buckets",
                "top_k_per_bucket": args.top_k_per_bucket,
                "similarity_threshold": args.similarity_threshold,
                "result_count": len(bucket_output["merged_results"]),
                **bucket_output,
            }
            write_json(rag_query_path, rag_query_output)
            evidence_path.write_text(
                format_bucket_evidence_markdown(bucket_output, args.max_chars_per_result),
                encoding="utf-8",
            )

            await run_retrieval(
                candidates_path=candidates_path,
                candidate_id=candidate_id,
                profile_paths=profile_paths,
                out_path=retrieval_path,
                extra_evidence_path=evidence_path,
            )
            await run_review(
                candidates_path=candidates_path,
                candidate_id=candidate_id,
                retrieval_path=retrieval_path,
                out_path=review_path,
            )

            retrieval = read_json(retrieval_path)
            review = read_json(review_path)
            item = {
                "candidate_id": candidate_id,
                "candidate_direction": candidate.get("candidate_direction"),
                "rag_result_count": rag_query_output["result_count"],
                "verification_conclusion": retrieval.get("verification_conclusion"),
                "claim_checks": retrieval.get("claim_checks"),
                "review_decision": review.get("review_decision"),
                "review_features": review.get("review_features"),
                "required_revision": review.get("required_revision"),
                "final_advice": review.get("final_advice"),
                "paths": paths,
            }
            return normalize_review_consistency(item)
        except Exception as exc:
            error_payload = {
                "candidate_id": candidate_id,
                "stage": "candidate_review_batch",
                "error": str(exc),
                "candidate": candidate,
                "paths": paths,
            }
            write_json(error_path, error_payload)
            return fallback_review_item(candidate, str(exc), paths)

    async def guarded_process(candidate: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_candidate(candidate)

    summary_items = await asyncio.gather(*(guarded_process(candidate) for candidate in candidates))

    summary = {
        "db_id": db_id,
        "candidate_count": len(summary_items),
        "top_k_per_bucket": args.top_k_per_bucket,
        "similarity_threshold": args.similarity_threshold,
        "items": summary_items,
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    write_json(summary_json, summary)
    summary_md.write_text(build_summary_markdown(summary), encoding="utf-8")

    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, ensure_ascii=False, indent=2))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="批量试跑 V2 候选方向 RAG 查证和可做性审查。")
    parser.add_argument("--existing-db-id", required=True, help="已建好的 Chroma db_id")
    parser.add_argument("--manifest", required=True, help="pdf_manifest.json 路径")
    parser.add_argument("--candidates", required=True, help="候选方向 JSON 数组路径")
    parser.add_argument("--profiles", nargs="+", required=True, help="候选生成时使用的论文画像路径列表")
    parser.add_argument("--candidate-ids", nargs="*", default=None, help="只处理指定 candidate_id；不填则处理全部")
    parser.add_argument("--out-dir", required=True, help="批量输出目录")
    parser.add_argument("--top-k-per-bucket", type=int, default=4, help="每桶召回条数；connection 多论文时会均衡分配")
    parser.add_argument("--similarity-threshold", type=float, default=0.0, help="相似度阈值")
    parser.add_argument("--max-chars-per-result", type=int, default=4500, help="Markdown 中每条证据最多保留字符数")
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
