"""选题建议系统 - 候选方向单次修改与轻量验收节点。

用途：读取单个候选方向和 review_agent 的 revise_keep 结果，
调用 proposal_agent 的 revise 模式改写该方向，再用轻量 guard 检查是否改坏。
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


REVISION_PROMPT = """你是科研候选方向修改助手。

你会收到一个原始候选方向、证据查证结果和可做性审查结果。

你的任务不是提出新方向，而是只根据审查结果中的 required_revision，对原候选方向做一次小幅收窄或改弱。

修改原则：
- 保留原 candidate_id。
- 保留或同步更新 candidate_title；candidate_title 必须是 12-18 个中文字符左右的短标题，不要写成长句。
- 保留原 base 论文和 module 论文，不要替换成新论文。
- 不要新增输入材料之外的新方法、数据集、实验结果或外部事实。
- 不要把研究假设写得更强，不要承诺一定提升。
- 优先把过宽表达改成局部替换、保留原结构项、小实验验证。
- questions_to_verify 要同步改成修改后方向最需要验证的问题。

输出必须是 JSON 对象，字段与原候选方向一致：

{
  "candidate_id": "C2",
  "candidate_title": "...",
  "candidate_direction": "...",
  "source_basis": {
    "base_paper": "...",
    "module_paper": "...",
    "combination_reason": "...",
    "compatibility_assumption": "..."
  },
  "research_hypothesis": "...",
  "research_value": "...",
  "questions_to_verify": ["..."]
}

要求：
- 只输出 JSON 对象，不要输出 Markdown、代码块或解释文字。
- 第一字符必须是 {。
- 所有自然语言内容使用中文。
"""


GUARD_PROMPT = """你是候选方向修改验收助手。

你会收到原始候选方向、修改后候选方向、审查结果和查证结果。

你的任务不是重新审查方向是否 keep/reject，而是检查修改有没有忠实执行 required_revision，且有没有改出幻觉。

只检查以下事项：
- keeps_original_base_and_module：是否保留原 base 和 module，没有替换成新论文。
- addresses_required_revision：是否回应了 required_revision 的核心修改要求。
- adds_no_new_paper_or_method：是否没有新增输入材料之外的新论文、新方法或外部事实。
- adds_no_unsupported_claim：是否没有新增查证结果不支持的强断言。
- does_not_strengthen_claim：是否没有把原方向说得更强或更确定。

guard_result 规则：
- pass：以上都满足。
- warning：基本满足，但仍有措辞偏强或修改不够彻底；可以使用修改版，但最终报告要保守表达。
- fail：新增幻觉、换了 base/module、没有回应 required_revision，或明显把结论说得更强。

输出必须是 JSON 对象：

{
  "candidate_id": "C2",
  "guard_result": "pass | warning | fail",
  "checked_items": {
    "keeps_original_base_and_module": true,
    "addresses_required_revision": true,
    "adds_no_new_paper_or_method": true,
    "adds_no_unsupported_claim": true,
    "does_not_strengthen_claim": true
  },
  "reason": "一句话说明验收结论"
}

要求：
- 只输出 JSON 对象，不要输出 Markdown、代码块或解释文字。
- 第一字符必须是 {。
- 所有自然语言内容使用中文。
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


def validate_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "candidate_id",
        "candidate_title",
        "candidate_direction",
        "source_basis",
        "research_hypothesis",
        "research_value",
        "questions_to_verify",
    }
    missing = required - set(candidate.keys())
    if missing:
        raise ValueError(f"修改后候选方向缺少字段: {sorted(missing)}")
    if not isinstance(candidate["source_basis"], dict):
        raise ValueError("source_basis 必须是对象")
    source_required = {
        "base_paper",
        "module_paper",
        "combination_reason",
        "compatibility_assumption",
    }
    source_missing = source_required - set(candidate["source_basis"].keys())
    if source_missing:
        raise ValueError(f"source_basis 缺少字段: {sorted(source_missing)}")
    if not isinstance(candidate["questions_to_verify"], list):
        raise ValueError("questions_to_verify 必须是数组")
    if not str(candidate.get("candidate_title", "")).strip():
        raise ValueError("candidate_title 不能为空")


def validate_guard(guard: dict[str, Any]) -> None:
    required = {"candidate_id", "guard_result", "checked_items", "reason"}
    missing = required - set(guard.keys())
    if missing:
        raise ValueError(f"guard 结果缺少字段: {sorted(missing)}")
    if guard["guard_result"] not in {"pass", "warning", "fail"}:
        raise ValueError("guard_result 枚举值不合法")
    if not isinstance(guard["checked_items"], dict):
        raise ValueError("checked_items 必须是对象")
    checked_required = {
        "keeps_original_base_and_module",
        "addresses_required_revision",
        "adds_no_new_paper_or_method",
        "adds_no_unsupported_claim",
        "does_not_strengthen_claim",
    }
    checked_missing = checked_required - set(guard["checked_items"].keys())
    if checked_missing:
        raise ValueError(f"checked_items 缺少字段: {sorted(checked_missing)}")


async def call_json_agent(name: str, system_message: str, task: str) -> tuple[dict[str, Any], str]:
    model_client = create_reading_model_client()
    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=system_message,
    )
    try:
        response = await agent.run(task=task)
        raw_content = str(response.messages[-1].content)
    finally:
        await model_client.close()
    return extract_json_object(raw_content), raw_content


async def run_revision(
    candidates_path: Path,
    candidate_id: str,
    review_path: Path,
    retrieval_path: Path,
    out_path: Path,
) -> None:
    original_candidate = find_candidate(load_candidates(candidates_path), candidate_id)
    review_result = json.loads(read_text(review_path))
    retrieval_result = json.loads(read_text(retrieval_path))

    if review_result.get("review_decision") != "revise_keep":
        final_package = {
            "candidate_id": candidate_id,
            "revision_applied": False,
            "final_source": "original_candidate",
            "original_candidate": original_candidate,
            "revised_candidate": None,
            "guard_result": None,
            "final_candidate": original_candidate,
            "review_result": review_result,
            "revision_advice": review_result.get("required_revision", ""),
            "note": "review_decision 不是 revise_keep，未触发自动修改。",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_package, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_path))
        return

    revision_task = "\n".join(
        [
            REVISION_PROMPT,
            "",
            "# original_candidate",
            json.dumps(original_candidate, ensure_ascii=False, indent=2),
            "",
            "# retrieval_result",
            json.dumps(retrieval_result, ensure_ascii=False, indent=2),
            "",
            "# review_result",
            json.dumps(review_result, ensure_ascii=False, indent=2),
            "",
        ]
    )
    revised_candidate, revision_raw = await call_json_agent(
        name="proposal_revision_probe_agent",
        system_message=(
            "你是严谨的科研候选方向修改助手。"
            "只能根据审查意见小幅修改原候选方向；最终回答只能是 JSON 对象。"
        ),
        task=revision_task,
    )
    validate_candidate(revised_candidate)

    guard_task = "\n".join(
        [
            GUARD_PROMPT,
            "",
            "# original_candidate",
            json.dumps(original_candidate, ensure_ascii=False, indent=2),
            "",
            "# revised_candidate",
            json.dumps(revised_candidate, ensure_ascii=False, indent=2),
            "",
            "# retrieval_result",
            json.dumps(retrieval_result, ensure_ascii=False, indent=2),
            "",
            "# review_result",
            json.dumps(review_result, ensure_ascii=False, indent=2),
            "",
        ]
    )
    guard_result, guard_raw = await call_json_agent(
        name="proposal_revision_guard_probe_agent",
        system_message=(
            "你是严谨的候选方向修改验收助手。"
            "只检查修改是否忠实和保守；最终回答只能是 JSON 对象。"
        ),
        task=guard_task,
    )
    validate_guard(guard_result)

    if guard_result["guard_result"] in {"pass", "warning"}:
        final_candidate = revised_candidate
        final_source = "revised_candidate"
        revision_applied = True
        note = "修改通过轻量验收，使用修改后候选方向。"
    else:
        final_candidate = original_candidate
        final_source = "original_candidate_with_revision_advice"
        revision_applied = False
        note = "修改未通过轻量验收，回退原始候选方向，并保留 review 修改建议。"

    final_package = {
        "candidate_id": candidate_id,
        "revision_applied": revision_applied,
        "final_source": final_source,
        "original_candidate": original_candidate,
        "revised_candidate": revised_candidate,
        "guard_result": guard_result,
        "final_candidate": final_candidate,
        "review_result": review_result,
        "revision_advice": review_result.get("required_revision", ""),
        "note": note,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".revision.raw.txt").write_text(revision_raw, encoding="utf-8")
    out_path.with_suffix(".guard.raw.txt").write_text(guard_raw, encoding="utf-8")
    out_path.write_text(json.dumps(final_package, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# 候选方向单次修改与验收结果",
                "",
                f"candidate_id: {candidate_id}",
                "",
                "## 最终采用",
                "",
                "```json",
                json.dumps(final_candidate, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Guard 结果",
                "",
                "```json",
                json.dumps(guard_result, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 修改建议",
                "",
                review_result.get("required_revision", ""),
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_path))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="试跑 V2 候选方向单次修改和轻量验收。")
    parser.add_argument("--candidates", required=True, help="候选方向 JSON 数组路径")
    parser.add_argument("--candidate-id", required=True, help="要修改的 candidate_id")
    parser.add_argument("--review", required=True, help="可做性审查 JSON 路径")
    parser.add_argument("--retrieval", required=True, help="证据查证 JSON 路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    asyncio.run(
        run_revision(
            candidates_path=Path(args.candidates),
            candidate_id=args.candidate_id,
            review_path=Path(args.review),
            retrieval_path=Path(args.retrieval),
            out_path=Path(args.out),
        )
    )


if __name__ == "__main__":
    main()
