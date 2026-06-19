"""选题建议系统 - 证据查证节点。

用途：读取候选方向 JSON 中的单个 candidate，结合论文画像材料，
调用大模型输出结构化查证结果。
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


RETRIEVAL_PROMPT = """你是科研证据查证助手。

你的任务是围绕一个候选选题，从提供的论文画像和 PDF/RAG 证据中查找支持证据、风险证据和证据缺口。

你不负责提出新选题，也不负责判断方向最终保留或驳回。

请重点查证：
1. base 论文是否真的存在候选方向中说的问题或遗留空间。
2. module 论文是否真的提供了候选方向中说的方法模块。
3. module 的输入、输出、依赖条件是否能和 base 问题接上。
4. 候选方向中的研究假设是否有材料支撑。
5. 是否存在任务设定、数据、监督信号或方法依赖上的风险。
6. 哪些关键判断当前材料还无法支持。

输出必须是 JSON 对象，字段如下：

{
  "candidate_id": "C1",
  "verification_target": "本次查证的候选方向和关键查证点",
  "claim_checks": {
    "base_problem": {
      "status": "supported | partially_supported | unsupported | contradicted",
      "evidence_refs": ["支撑该判断的 evidence_source"],
      "note": "base 论文是否真的存在候选方向所说的问题、遗留空间或可改进点"
    },
    "module_method": {
      "status": "supported | partially_supported | unsupported | contradicted",
      "evidence_refs": ["支撑该判断的 evidence_source"],
      "note": "module 论文是否真的提供了候选方向所说的可迁移方法模块"
    },
    "connection": {
      "status": "supported | partially_supported | unsupported | contradicted",
      "evidence_refs": ["支撑该判断的 evidence_source"],
      "note": "module 的输入、输出、依赖条件是否能接到 base 问题"
    },
    "key_assumptions": [
      {
        "assumption": "候选方向成立所依赖的关键前提",
        "status": "supported | unverified | contradicted",
        "evidence_refs": ["相关 evidence_source"],
        "note": "该前提目前被材料支持、未验证或被反证"
      }
    ]
  },
  "supporting_evidence": [
    {
      "supported_claim": "被支持的判断",
      "evidence_source": "证据来源，例如 paper_id / 画像 / section / page",
      "evidence_summary": "证据大意",
      "evidence_strength": "strong | medium | weak"
    }
  ],
  "risk_evidence": [
    {
      "risk_point": "风险点",
      "evidence_source": "证据来源",
      "evidence_summary": "证据大意",
      "risk_strength": "high | medium | low"
    }
  ],
  "evidence_gaps": [
    "当前材料无法支持或需要继续精读确认的关键点"
  ],
  "verification_conclusion": "sufficient | partially_sufficient | insufficient | contradicted",
  "conclusion_reason": "2-4 句话说明材料层面的查证结论"
}

要求：
- 只输出 JSON 对象，不要输出 Markdown、代码块或解释文字。
- 第一字符必须是 {。
- 所有自然语言内容使用中文。
- 所有证据必须来自输入材料。
- claim_checks 只做材料分桶，不输出最终保留、修改后保留、暂缓或驳回。
- 如果没有明确风险证据，risk_evidence 输出空数组，并在 evidence_gaps 或 conclusion_reason 中说明不代表方向已成立。
- 不要输出“保留 / 修改后保留 / 暂缓 / 驳回”，这是 review_agent 的职责。
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_profile(path: Path) -> str:
    content = read_text(path)
    marker = "## 模型输出"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content.strip()


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
        "verification_target",
        "claim_checks",
        "supporting_evidence",
        "risk_evidence",
        "evidence_gaps",
        "verification_conclusion",
        "conclusion_reason",
    }
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"查证结果缺少字段: {sorted(missing)}")
    if result["verification_conclusion"] not in {
        "sufficient",
        "partially_sufficient",
        "insufficient",
        "contradicted",
    }:
        raise ValueError("verification_conclusion 枚举值不合法")
    if not isinstance(result["claim_checks"], dict):
        raise ValueError("claim_checks 必须是对象")
    required_checks = {"base_problem", "module_method", "connection", "key_assumptions"}
    missing_checks = required_checks - set(result["claim_checks"].keys())
    if missing_checks:
        raise ValueError(f"claim_checks 缺少字段: {sorted(missing_checks)}")
    if not isinstance(result["supporting_evidence"], list):
        raise ValueError("supporting_evidence 必须是数组")
    if not isinstance(result["risk_evidence"], list):
        raise ValueError("risk_evidence 必须是数组")
    if not isinstance(result["evidence_gaps"], list):
        raise ValueError("evidence_gaps 必须是数组")


async def run_retrieval(
    candidates_path: Path,
    candidate_id: str,
    profile_paths: list[Path],
    out_path: Path,
    extra_evidence_path: Path | None,
) -> None:
    candidate = find_candidate(load_candidates(candidates_path), candidate_id)
    profiles = []
    for index, path in enumerate(profile_paths, start=1):
        profiles.append(
            "\n".join(
                [
                    f"# paper_profile_{index}",
                    f"source_file: {path.name}",
                    "",
                    read_profile(path),
                    "",
                ]
            )
        )

    extra_evidence = ""
    if extra_evidence_path:
        extra_evidence = "\n".join([
            "# extra_pdf_or_rag_evidence",
            read_text(extra_evidence_path),
            "",
        ])

    task = "\n".join(
        [
            RETRIEVAL_PROMPT,
            "",
            "# candidate_to_verify",
            json.dumps(candidate, ensure_ascii=False, indent=2),
            "",
            "# available_paper_profiles",
            "\n".join(profiles),
            "",
            extra_evidence,
        ]
    )

    model_client = create_reading_model_client()
    agent = AssistantAgent(
        name="proposal_retrieval_probe_agent",
        model_client=model_client,
        system_message=(
            "你是严谨的科研证据查证助手。只能基于输入材料查找证据。"
            "不要输出 Markdown、代码块或解释文字；最终回答只能是 JSON 对象。"
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
                "# 证据查证 Agent 试跑结果",
                "",
                f"candidate_id: {candidate_id}",
                "",
                "## 输入候选方向",
                "",
                "```json",
                json.dumps(candidate, ensure_ascii=False, indent=2),
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

    parser = argparse.ArgumentParser(description="试跑 V2 证据查证 Agent。")
    parser.add_argument("--candidates", required=True, help="候选方向 JSON 数组路径")
    parser.add_argument("--candidate-id", required=True, help="要查证的 candidate_id")
    parser.add_argument("--profiles", nargs="+", required=True, help="论文画像 Markdown 路径列表")
    parser.add_argument("--extra-evidence", default=None, help="可选 PDF/RAG 额外证据文本路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    asyncio.run(
        run_retrieval(
            candidates_path=Path(args.candidates),
            candidate_id=args.candidate_id,
            profile_paths=[Path(path) for path in args.profiles],
            out_path=Path(args.out),
            extra_evidence_path=Path(args.extra_evidence) if args.extra_evidence else None,
        )
    )


if __name__ == "__main__":
    main()
