"""选题建议系统 - 候选方向批量单次修改节点。

用途：根据 review 结果批量执行 revise_keep 的单次修改和 guard，
并按 keep / revise_keep / defer / reject 分流生成最终候选列表。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.agents.proposal_pipeline.revision_node import run_revision


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_candidate(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ValueError(f"未找到候选方向: {candidate_id}")


def build_markdown_report(
    final_candidates: list[dict[str, Any]],
    deferred_candidates: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "# 候选方向单次修改批量结果",
        "",
        "## 最终候选方向",
        "",
    ]
    for item in final_candidates:
        candidate = item["final_candidate"]
        lines.extend(
            [
                f"### {item['candidate_id']} | {item['review_decision']} | {item['final_source']}",
                "",
                f"- 候选方向：{candidate.get('candidate_direction', '')}",
                f"- guard：{item.get('guard_result') or '未触发'}",
                f"- 建议：{item.get('final_advice', '')}",
                "",
            ]
        )

    lines.extend(["## 暂缓候选方向", ""])
    for item in deferred_candidates:
        lines.extend(
            [
                f"### {item['candidate_id']} | defer",
                "",
                f"- 原方向：{item['candidate'].get('candidate_direction', '')}",
                f"- 原因：{item.get('final_advice', '')}",
                "",
            ]
        )

    lines.extend(["## 拒绝候选方向", ""])
    for item in rejected_candidates:
        lines.extend(
            [
                f"### {item['candidate_id']} | reject",
                "",
                f"- 原方向：{item['candidate'].get('candidate_direction', '')}",
                f"- 原因：{item.get('final_advice', '')}",
                "",
            ]
        )
    return "\n".join(lines)


async def run_batch(args: argparse.Namespace) -> None:
    candidates_path = Path(args.candidates)
    summary_path = Path(args.review_summary)
    review_dir = Path(args.review_dir)
    out_dir = Path(args.out_dir)

    candidates = read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("候选方向文件必须是 JSON 数组")
    summary = read_json(summary_path)
    items = summary.get("items") or []
    out_dir.mkdir(parents=True, exist_ok=True)

    final_candidates: list[dict[str, Any]] = []
    deferred_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    internal_records: list[dict[str, Any]] = []

    for item in items:
        candidate_id = str(item.get("candidate_id"))
        candidate = find_candidate(candidates, candidate_id)
        review_decision = item.get("review_decision")
        review_path = review_dir / f"{candidate_id}_review.json"
        retrieval_path = review_dir / f"{candidate_id}_retrieval.json"

        if review_decision in {"keep", "revise_keep"}:
            revision_path = out_dir / f"{candidate_id}_revision.json"
            revision_error_path = out_dir / f"{candidate_id}_revision_error.json"
            revision_failed = None
            try:
                await run_revision(
                    candidates_path=candidates_path,
                    candidate_id=candidate_id,
                    review_path=review_path,
                    retrieval_path=retrieval_path,
                    out_path=revision_path,
                )
            except Exception as exc:
                revision_failed = str(exc)
                fallback_package = {
                    "candidate_id": candidate_id,
                    "revision_applied": False,
                    "final_source": "original_candidate_with_revision_error",
                    "original_candidate": candidate,
                    "revised_candidate": None,
                    "guard_result": {
                        "candidate_id": candidate_id,
                        "guard_result": "fail",
                        "checked_items": {},
                        "reason": f"单次修改失败，已回退原候选：{revision_failed}",
                    },
                    "final_candidate": candidate,
                    "review_result": read_json(review_path) if review_path.exists() else {},
                    "revision_advice": item.get("required_revision", ""),
                    "note": "单次修改失败，系统兜底回退原始候选方向。",
                    "business_fallback": {
                        "type": "revision_error_fallback",
                        "reason": revision_failed,
                    },
                }
                write_json(revision_path, fallback_package)
                write_json(
                    revision_error_path,
                    {
                        "candidate_id": candidate_id,
                        "stage": "revision_batch",
                        "error": revision_failed,
                        "candidate": candidate,
                        "review_path": str(review_path),
                        "retrieval_path": str(retrieval_path),
                    },
                )
            revision_package = read_json(revision_path)
            guard = revision_package.get("guard_result") or {}
            final_item = {
                "candidate_id": candidate_id,
                "review_decision": review_decision,
                "final_source": revision_package.get("final_source"),
                "guard_result": guard.get("guard_result") if guard else None,
                "final_candidate": revision_package.get("final_candidate"),
                "required_revision": item.get("required_revision"),
                "final_advice": item.get("final_advice"),
                "paths": {
                    "revision": str(revision_path),
                    "revision_error": str(revision_error_path) if revision_failed else None,
                    "review": str(review_path),
                    "retrieval": str(retrieval_path),
                },
            }
            if revision_failed:
                final_item["business_fallback"] = {
                    "type": "revision_error_fallback",
                    "reason": revision_failed,
                }
            final_candidates.append(final_item)
            internal_records.append({**final_item, "revision_package": revision_package})
        elif review_decision == "defer":
            deferred_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "review_decision": review_decision,
                    "final_advice": item.get("final_advice"),
                    "review_features": item.get("review_features"),
                    "paths": {
                        "review": str(review_path),
                        "retrieval": str(retrieval_path),
                    },
                }
            )
        elif review_decision == "reject":
            rejected_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "review_decision": review_decision,
                    "final_advice": item.get("final_advice"),
                    "review_features": item.get("review_features"),
                    "paths": {
                        "review": str(review_path),
                        "retrieval": str(retrieval_path),
                    },
                }
            )
        else:
            raise ValueError(f"{candidate_id} 的 review_decision 不合法: {review_decision}")

    final_output = {
        "final_candidate_count": len(final_candidates),
        "deferred_candidate_count": len(deferred_candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "final_candidates": final_candidates,
        "deferred_candidates": deferred_candidates,
        "rejected_candidates": rejected_candidates,
    }
    internal_output = {
        **final_output,
        "internal_records": internal_records,
    }

    write_json(out_dir / "final_candidates_probe.json", final_output)
    write_json(out_dir / "deferred_candidates_probe.json", deferred_candidates)
    write_json(out_dir / "rejected_candidates_probe.json", rejected_candidates)
    write_json(out_dir / "revision_batch_internal.json", internal_output)
    (out_dir / "final_candidates_probe.md").write_text(
        build_markdown_report(final_candidates, deferred_candidates, rejected_candidates),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "final_candidates": str(out_dir / "final_candidates_probe.json"),
                "deferred_candidates": str(out_dir / "deferred_candidates_probe.json"),
                "markdown": str(out_dir / "final_candidates_probe.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="批量试跑 V2 候选方向单次修改和分流。")
    parser.add_argument("--candidates", required=True, help="候选方向 JSON 数组路径")
    parser.add_argument("--review-summary", required=True, help="候选批量审查 summary.json 路径")
    parser.add_argument("--review-dir", required=True, help="包含 Cx_retrieval.json / Cx_review.json 的目录")
    parser.add_argument("--out-dir", required=True, help="批量修改输出目录")
    args = parser.parse_args()

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
