"""选题建议系统 - 论文画像节点。

用途：从已有 pdf_manifest 和 PDF 章节切分结果中构造单篇论文证据包，
调用大模型生成论文画像 Markdown。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent

from src.agents.proposal_pipeline.pdf_parse import ParsedSection, extract_sections, split_section
from src.core.model_client import create_reading_model_client


PAPER_PROFILE_PROMPT = """你是一名科研选题分析助手。

你的任务不是普通论文总结，而是把单篇论文分析成一张“选题画像卡片”，供后续 Agent 判断这篇论文能否作为 base、能否提供 module、是否能启发新的研究方向。

你会收到：
1. arXiv 元数据：标题、作者、年份、摘要、检索角色、命中的检索词。
2. PDF 正文证据包：按章节提供的 Introduction、Method、Experiment、Conclusion 等片段。

请严格基于提供的材料分析，不要编造论文中没有出现的信息。
如果证据不足，请明确说明。

不要输出寒暄、确认、解释自己身份或任务说明。最终回答必须直接从“## 解决的问题”开始。

请按下面的阅读流程完成分析。

第一步：先读 arXiv 摘要和 Introduction。
目标：判断这篇论文解决的问题。
请回答：这篇论文研究什么问题？这个问题发生在什么任务场景中？为什么这个问题值得解决？现有方法或现有设定中有什么不足，导致作者要做这篇论文？

第二步：再读 Method、Experiment、Conclusion。
目标：同时判断核心方法和遗留问题。
请先分析核心方法：论文最关键的 1-3 个方法抓手是什么？每个方法的输入是什么、输出是什么、依赖条件是什么、为什么它在原论文中能起作用？
核心方法只提取能被后续研究借用、替换或迁移的方法模块；理论分析、证明和实验结论不要作为核心方法，除非它本身能形成可迁移的方法工具。
然后分析遗留问题：结论中是否明确提到 future work 或后续方向？方法本身依赖了什么假设、输入条件、模块设计或适用场景？实验结果是否暴露了某些短板、未覆盖场景或可继续分析的地方？哪些地方可能成为后续改进切入口？
注意：大多数论文不会主动写自己的局限，所以你可以基于方法结构推断潜在遗留问题。但如果是推断，必须明确写“这是基于方法结构的推断”，不能写成作者明确承认。

第三步：基于前两步，判断可利用方式。
目标：判断这篇论文在“模块迁移式选题”中怎么用。
请明确判断：适合作为 base、适合作为 module、两者都适合，或暂不适合。
如果适合作为 base，说明它最值得被改进的点是什么。
如果适合作为 module，说明哪个方法模块值得迁移、迁移前提是什么、直接迁移有什么风险。
不要只写“有参考价值”，必须说明具体怎么用。

第四步：整理支撑证据。
目标：把前面关键判断和论文证据绑定起来。
每条证据按这个格式写：
- 支撑判断：
- 证据位置：
- 证据大意：
- 证据强度：强 / 中 / 弱

要求：
- 每条证据必须支撑一个具体判断。
- 不要泛泛写“来自方法部分”。
- 不要引用没有提供的内容。
- 如果某个判断证据不足，要明确说明。

最终输出前，请在内部做一次轻量自检，但不要输出自检过程：
1. 核心方法是否都是可借用、可替换或可迁移的方法模块。
2. 遗留问题是否区分了明确遗留问题和基于方法结构的推断。
3. 可利用方式是否能从解决的问题、核心方法或遗留问题推出。
4. 支撑证据是否覆盖了主要关键判断。
5. 如果某个判断证据不足，是否已经明确标注。

最终请按以下五个栏目输出：

## 解决的问题
输出第一步的结论。
长度限制：1 段，最多 4 句话。

## 核心方法
输出第二步中关于核心方法的结论。
长度限制：列 1-3 个方法；每个方法最多 5 行。

## 遗留问题
输出第二步中关于遗留问题的结论，并区分“明确遗留问题”和“基于方法结构的推断”。
长度限制：列 2-4 条。

## 可利用方式
输出第三步的判断。
长度限制：先给一个明确角色判断，再列 2-4 条理由、迁移前提或迁移风险。

## 支撑证据
输出第四步整理的证据。
长度限制：列 4-6 条，每条绑定一个具体判断。

整体要求：
1. 使用中文。
2. 使用 Markdown。
3. 不要输出 JSON。
4. 不要写成长篇综述。
5. 重点服务后续选题生成。
6. 对不确定内容要保守表达。
"""


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_paper(manifest: dict[str, Any], paper_id: str | None) -> dict[str, Any]:
    papers = [
        paper
        for paper in manifest.get("papers", [])
        if paper.get("pdf_status") in ("cached", "downloaded") and paper.get("pdf_path")
    ]
    if not papers:
        raise ValueError("没有可用 PDF 论文")
    if paper_id is None:
        return papers[0]
    for paper in papers:
        if paper.get("paper_id") == paper_id:
            return paper
    raise ValueError(f"未找到可用论文: {paper_id}")


def select_sections(sections: list[ParsedSection]) -> list[ParsedSection]:
    """按论文画像需要选择证据章节。"""
    selected_keys = {
        "abstract",
        "introduction",
        "method",
        "experiment",
        "discussion",
        "conclusion",
    }
    selected: list[ParsedSection] = []
    for section in sections:
        if section.section_key not in selected_keys:
            continue
        if section.section_key == "experiment":
            title_lower = section.section_title.lower()
            useful_keywords = ["result", "ablation", "analysis", "discussion"]
            low_value_keywords = ["setup", "dataset", "implementation"]
            if any(keyword in title_lower for keyword in low_value_keywords) and not any(
                keyword in title_lower for keyword in useful_keywords
            ):
                continue
        selected.append(section)
    return selected


def build_evidence_pack(paper: dict[str, Any], max_chars_per_section: int) -> str:
    sections = select_sections(extract_sections(Path(paper["pdf_path"])))
    lines = [
        "# arXiv 元数据",
        "",
        f"- paper_id：{paper.get('paper_id')}",
        f"- title：{paper.get('title')}",
        f"- year：{paper.get('published')}",
        f"- authors：{', '.join(paper.get('authors') or [])}",
        f"- roles：{', '.join(paper.get('roles') or [])}",
        f"- matched_queries：{', '.join(paper.get('matched_queries') or [])}",
        "",
        "## arXiv 摘要",
        paper.get("summary") or "",
        "",
        "# PDF 正文证据包",
        "",
    ]

    for section in sections:
        chunks = split_section(section, max_chars=max_chars_per_section, overlap=0)
        for chunk_index, chunk in enumerate(chunks[:2]):
            lines.extend(
                [
                    (
                        f"## section_key={section.section_key}; "
                        f"section_title={section.section_title}; "
                        f"pages={section.page_start}-{section.page_end}; "
                        f"chunk_index={chunk_index}"
                    ),
                    "",
                    chunk[:max_chars_per_section],
                    "",
                ]
            )

    return "\n".join(lines)


async def run_profile(manifest_path: Path, paper_id: str | None, out_path: Path, max_chars_per_section: int) -> None:
    manifest = load_manifest(manifest_path)
    paper = find_paper(manifest, paper_id)
    evidence_pack = build_evidence_pack(paper, max_chars_per_section)
    task = f"{PAPER_PROFILE_PROMPT}\n\n下面是待分析论文材料：\n\n{evidence_pack}"

    model_client = create_reading_model_client()
    agent = AssistantAgent(
        name="proposal_paper_profile_probe_agent",
        model_client=model_client,
        system_message="你是严谨的科研选题分析助手，只能基于用户提供的论文材料分析。",
    )
    try:
        response = await agent.run(task=task)
        content = str(response.messages[-1].content)
    finally:
        await model_client.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "# 论文画像 Agent 试跑结果",
                "",
                f"paper_id：{paper.get('paper_id')}",
                f"title：{paper.get('title')}",
                "",
                "## 输入证据包",
                "",
                "```markdown",
                evidence_pack,
                "```",
                "",
                "## 模型输出",
                "",
                content,
            ]
        ),
        encoding="utf-8",
    )
    print(str(out_path))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="试跑 V2 论文画像 Agent。")
    parser.add_argument("--manifest", required=True, help="pdf_manifest.json 路径")
    parser.add_argument("--paper-id", default=None, help="指定 paper_id；不填则取第一篇可用 PDF")
    parser.add_argument("--out", required=True, help="输出 Markdown 路径")
    parser.add_argument("--max-chars-per-section", type=int, default=3500, help="每个章节 chunk 最大字符数")
    args = parser.parse_args()

    asyncio.run(
        run_profile(
            manifest_path=Path(args.manifest),
            paper_id=args.paper_id,
            out_path=Path(args.out),
            max_chars_per_section=args.max_chars_per_section,
        )
    )


if __name__ == "__main__":
    main()
