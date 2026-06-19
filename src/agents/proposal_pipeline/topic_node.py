"""选题建议系统 - 候选选题生成节点。

用途：读取多篇论文画像 Markdown，调用大模型生成候选研究方向。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from src.core.model_client import create_reading_model_client
from src.utils.llm_json import LLMJsonParseError, parse_llm_json_array


TOPIC_PROPOSAL_PROMPT = """你是一名科研选题构思助手。

你的任务不是总结论文，也不是自由发散选题，而是基于多篇“论文画像卡片”做 base-module 匹配，生成少量值得继续精读和验证的候选研究方向。

你会收到多篇论文画像。每篇画像包含：
- 解决的问题
- 核心方法
- 遗留问题
- 可利用方式
- 支撑证据

请严格基于输入材料，不要编造新的论文、外部事实或不存在的方法模块。

请按以下逻辑思考：

1. 先找适合作为 base 的论文：
   重点看它是否有清晰问题、遗留问题、方法依赖、适用边界或可改进点。

2. 再找适合作为 module 的论文：
   重点看它是否有可迁移的方法模块，以及该模块的输入、输出、依赖条件和迁移风险。

3. 再判断 base-module 是否能组合：
   - base 的遗留问题需要什么能力？
   - module 是否提供这种能力？
   - module 的输入输出能否接到 base 的问题场景？
   - 二者在任务、数据、监督信号或模型结构上是否基本兼容？

只有存在明确连接点时，才生成候选方向。
不要生成“两个论文都相关，所以可以组合”的方向。

请生成 3-5 个候选方向。如果高质量方向不足，可以少于 3 个，不要硬凑。

输出必须是 JSON 数组。每个候选方向是一个独立对象，字段如下：

[
  {
    "candidate_id": "C1",
    "candidate_title": "12-18 个中文字符的短标题，像论文小标题一样概括方向，不要写成长句，不要出现“论文1/论文2”",
    "candidate_direction": "用一句话说明这个方向要做什么",
    "source_basis": {
      "base_paper": "base 论文提供了什么问题或遗留空间",
      "module_paper": "module 论文提供了什么可迁移方法",
      "combination_reason": "为什么这个方法可能作用于这个问题",
      "compatibility_assumption": "二者需要满足什么兼容性前提"
    },
    "research_hypothesis": "方向性研究假设：如果把某个 module 引入某个 base 问题，可能因为什么机制带来什么改进",
    "research_value": "说明这个方向为什么值得继续看",
    "questions_to_verify": [
      "用户后续精读时必须确认的问题"
    ]
  }
]

输出要求：
- 不要输出 Markdown。
- 不要输出代码块标记。
- 不要寒暄、确认、说明你将如何执行，也不要解释任务。
- 最终回答只能包含 JSON 数组，第一字符必须是 [。
- candidate_id 使用 C1、C2、C3...。
- candidate_title 必须是短标题，不要超过 18 个中文字符；不要截断 candidate_direction；不要写“候选研究方向”。
- 每个候选方向对象内部的自然语言内容使用中文。
- 最多输出 5 个方向。
- 所有判断必须基于输入论文画像。
- research_hypothesis 只给方向性假设，不要给出具体提升幅度、百分比或数值结果。
"""


def read_profile(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    marker = "## 模型输出"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content.strip()


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """从模型输出中提取 JSON 数组。"""
    try:
        value = parse_llm_json_array(text)
    except LLMJsonParseError as exc:
        raise ValueError("模型输出中未找到 JSON 数组") from exc

    if not isinstance(value, list):
        raise ValueError("模型输出不是 JSON 数组")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("JSON 数组元素必须是对象")
    return value


def validate_candidates(candidates: list[dict[str, Any]]) -> None:
    required = {
        "candidate_id",
        "candidate_title",
        "candidate_direction",
        "source_basis",
        "research_hypothesis",
        "research_value",
        "questions_to_verify",
    }
    source_required = {
        "base_paper",
        "module_paper",
        "combination_reason",
        "compatibility_assumption",
    }
    for index, item in enumerate(candidates, start=1):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"候选方向 {index} 缺少字段: {sorted(missing)}")
        if not isinstance(item["source_basis"], dict):
            raise ValueError(f"候选方向 {index} 的 source_basis 必须是对象")
        source_missing = source_required - set(item["source_basis"].keys())
        if source_missing:
            raise ValueError(f"候选方向 {index} 的 source_basis 缺少字段: {sorted(source_missing)}")
        if not isinstance(item["questions_to_verify"], list):
            raise ValueError(f"候选方向 {index} 的 questions_to_verify 必须是数组")
        if not str(item.get("candidate_title", "")).strip():
            raise ValueError(f"候选方向 {index} 的 candidate_title 不能为空")


async def run_topic(profile_paths: list[Path], out_path: Path) -> None:
    profiles = []
    for index, path in enumerate(profile_paths, start=1):
        profiles.append(
            "\n".join(
                [
                    f"# 论文画像 {index}",
                    f"来源文件：{path.name}",
                    "",
                    read_profile(path),
                    "",
                ]
            )
        )
    task = f"{TOPIC_PROPOSAL_PROMPT}\n\n下面是多篇论文画像：\n\n{''.join(profiles)}"

    model_client = create_reading_model_client()
    agent = AssistantAgent(
        name="proposal_topic_probe_agent",
        model_client=model_client,
        system_message=(
            "你是严谨的科研选题构思助手，只能基于用户提供的论文画像生成候选方向。"
            "不要输出任何寒暄、确认、过程说明、任务解释、Markdown 或代码块；最终回答只能是 JSON 数组。"
        ),
    )
    try:
        response = await agent.run(task=task)
        raw_content = str(response.messages[-1].content)
    finally:
        await model_client.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.txt")
    raw_path.write_text(raw_content, encoding="utf-8")

    candidates = extract_json_array(raw_content)
    validate_candidates(candidates)

    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    debug_path = out_path.with_suffix(".md")
    debug_path.write_text(
        "\n".join(
            [
                "# 候选选题生成 Agent 试跑结果",
                "",
                "## 输入画像文件",
                "",
                *[f"- {path}" for path in profile_paths],
                "",
                "## 模型输出",
                "",
                "```json",
                json.dumps(candidates, ensure_ascii=False, indent=2),
                "```",
                "",
                "## 原始输出",
                "",
                raw_content,
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_path))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="试跑 V2 候选选题生成 Agent。")
    parser.add_argument("--profiles", nargs="+", required=True, help="论文画像 Markdown 路径列表")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    asyncio.run(
        run_topic(
            profile_paths=[Path(path) for path in args.profiles],
            out_path=Path(args.out),
        )
    )


if __name__ == "__main__":
    main()
