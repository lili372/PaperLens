"""选题建议系统 - 可做性审查节点。

用途：读取单个候选方向和对应证据查证结果，调用大模型输出审查结论。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_reading_model_client
from src.utils.llm_json import LLMJsonParseError, parse_llm_json_object


REVIEW_PROMPT = """你是科研选题可做性审查助手。

你的任务是基于一个候选方向和证据查证结果，判断该方向如何处理。

你不是选题生成者，不要提出全新方向；你不是证据查证者，不要编造输入材料之外的新证据。

核心原则：不要凭感觉审查。必须先从查证结果中抽取客观特征，再按规则映射结论。

请先判断以下特征：
- base_problem_status：base 论文的问题、遗留空间或可改进点是否有证据支持。枚举值：supported | partially_supported | unsupported | contradicted。
- module_status：module 论文的方法模块是否真实存在，且确实可作为方法模块。枚举值：supported | partially_supported | unsupported | contradicted。
- connection_status：module 的输入、输出、依赖条件是否能接到 base 问题。枚举值：supported | partially_supported | unsupported | contradicted。
- dependency_conflict：是否存在任务设定、数据、监督信号、优化方式或输入输出上的明显冲突。
- critical_evidence_gap_count：会直接影响方向能否成立的关键证据缺口数量。
- major_risk_count：高风险或足以改变方向可做性的风险数量。
- unsupported_assumption_count：候选方向中关键但当前未被材料支持的前提数量。

结论只能是以下四个之一：
- keep：base_problem_status、module_status、connection_status 都是 supported；没有明显依赖冲突；没有关键证据缺口；没有高风险。
- revise_keep：base_problem_status 和 module_status 是 supported 或 partially_supported，connection_status 是 supported 或 partially_supported，且 dependency_conflict=false；方向表达过强、范围过宽、迁移前提不清，或风险可通过收窄方向、保留原结构项、改成局部替换/小实验验证来处理。
- defer：base/module/connection 中存在 unsupported，但没有明确反证或硬冲突；当前材料无法判断方向是否值得探索，必须补充额外论文、外部事实或关键 PDF 证据。
- reject：base/module/connection 中存在 contradicted，或 module 不是可迁移方法，或存在明显依赖冲突/反证，导致方向基本不值得继续探索。

本系统的目标不是证明研究方向一定成立，而是帮研究生筛掉明显不值得探索的方向，并保留值得继续精读和小实验验证的方向。
因此，存在“需要实验验证”的前提不等于 defer。
如果 base/module/connection 已基本成立（supported 或 partially_supported），且 dependency_conflict=false 且 major_risk_count=0，即使还有待验证前提，也优先判 revise_keep，并在 required_revision 中收窄方向。
只有当缺口会导致当前无法判断方向是否有探索价值时，才判 defer。
如果 dependency_conflict=true，不允许输出 keep 或 revise_keep；应根据冲突强度输出 defer 或 reject。

输出必须是 JSON 对象，字段如下：

{
  "candidate_id": "C1",
  "review_features": {
    "base_problem_status": "supported | partially_supported | unsupported | contradicted",
    "module_status": "supported | partially_supported | unsupported | contradicted",
    "connection_status": "supported | partially_supported | unsupported | contradicted",
    "dependency_conflict": false,
    "critical_evidence_gap_count": 0,
    "major_risk_count": 0,
    "unsupported_assumption_count": 0
  },
  "review_decision": "keep | revise_keep | defer | reject",
  "decision_reason": "2-4 句话说明为什么这些特征映射到该结论",
  "required_revision": "如果是 revise_keep，说明必须如何收窄或改弱；否则写空字符串",
  "validation_path": "用户后续精读或实验时应优先验证什么",
  "final_advice": "给用户的一句话建议"
}

要求：
- 只输出 JSON 对象，不要输出 Markdown、代码块或解释文字。
- 第一字符必须是 {。
- 所有自然语言内容使用中文。
- 只能使用输入的候选方向和查证结果。
- 不要因为方向有价值就默认 revise_keep；如果缺口必须靠补充材料解决，应输出 defer。
- 不要把 evidence_gaps 的普通待验证问题都算成 critical_evidence_gap，只有会影响方向是否成立的才计入。
- 不要把“后续需要做实验验证”的正常科研不确定性直接判为 defer；能通过收窄方向形成可探索小实验的，应判 revise_keep。
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_candidates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(read_text(path))
    if not isinstance(data, list):
        raise ValueError("候选方向文件必须是 JSON 数组")
    return data


def find_candidate(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ValueError(f"未找到候选方向: {candidate_id}")


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = parse_llm_json_object(text)
    except LLMJsonParseError as exc:
        raise ValueError("模型输出中未找到 JSON 对象") from exc

    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return value


def validate_result(result: dict[str, Any]) -> None:
    required = {
        "candidate_id",
        "review_features",
        "review_decision",
        "decision_reason",
        "required_revision",
        "validation_path",
        "final_advice",
    }
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"审查结果缺少字段: {sorted(missing)}")
    if result["review_decision"] not in {"keep", "revise_keep", "defer", "reject"}:
        raise ValueError("review_decision 枚举值不合法")
    if not isinstance(result["review_features"], dict):
        raise ValueError("review_features 必须是对象")

    feature_required = {
        "base_problem_status",
        "module_status",
        "connection_status",
        "dependency_conflict",
        "critical_evidence_gap_count",
        "major_risk_count",
        "unsupported_assumption_count",
    }
    feature_missing = feature_required - set(result["review_features"].keys())
    if feature_missing:
        raise ValueError(f"review_features 缺少字段: {sorted(feature_missing)}")
    status_values = {"supported", "partially_supported", "unsupported", "contradicted"}
    for key in {"base_problem_status", "module_status", "connection_status"}:
        if result["review_features"].get(key) not in status_values:
            raise ValueError(f"{key} 枚举值不合法")

    if result["review_features"].get("dependency_conflict") is True and result["review_decision"] in {
        "keep",
        "revise_keep",
    }:
        raise ValueError("dependency_conflict=true 时 review_decision 不能是 keep 或 revise_keep")


async def run_review(
    candidates_path: Path,
    candidate_id: str,
    retrieval_path: Path,
    out_path: Path,
) -> None:
    candidate = find_candidate(load_candidates(candidates_path), candidate_id)
    retrieval_result = json.loads(read_text(retrieval_path))

    task = "\n".join(
        [
            REVIEW_PROMPT,
            "",
            "# candidate_to_review",
            json.dumps(candidate, ensure_ascii=False, indent=2),
            "",
            "# retrieval_result",
            json.dumps(retrieval_result, ensure_ascii=False, indent=2),
            "",
        ]
    )

    model_client = create_reading_model_client()
    agent = AssistantAgent(
        name="proposal_review_probe_agent",
        model_client=model_client,
        system_message=(
            "你是严谨的科研选题可做性审查助手。"
            "只能基于输入材料审查；最终回答只能是 JSON 对象。"
        ),
    )
    try:
        response = await agent.run(task=task)
        raw_content = str(response.messages[-1].content)
    finally:
        await model_client.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".raw.txt").write_text(raw_content, encoding="utf-8")

    result = extract_json_object(raw_content)
    validate_result(result)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# 可做性审查 Agent 试跑结果",
                "",
                f"candidate_id: {candidate_id}",
                "",
                "## 输入候选方向",
                "",
                "```json",
                json.dumps(candidate, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 输入查证结果",
                "",
                "```json",
                json.dumps(retrieval_result, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 模型输出",
                "",
                "```json",
                json.dumps(result, ensure_ascii=False, indent=2),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_path))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="试跑 V2 可做性审查 Agent。")
    parser.add_argument("--candidates", required=True, help="候选方向 JSON 数组路径")
    parser.add_argument("--candidate-id", required=True, help="要审查的 candidate_id")
    parser.add_argument("--retrieval", required=True, help="证据查证 JSON 路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    asyncio.run(
        run_review(
            candidates_path=Path(args.candidates),
            candidate_id=args.candidate_id,
            retrieval_path=Path(args.retrieval),
            out_path=Path(args.out),
        )
    )


if __name__ == "__main__":
    main()
