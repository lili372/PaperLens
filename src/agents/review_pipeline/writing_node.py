"""综述系统 - 写作节点（合并 聚类后的 分析+写作 为一步）

输入：extracted_data（抽取的论文）+ user_request
处理：
  1. 聚类（算法，复用 PaperClusterAgent）：论文按研究方向分组，每组带主题名
  2. 写作（一个 LLM）：按方向分组的证据 + 用户请求 → 直接写出整篇综述
输出：review_markdown（整篇综述全文，含 [来源: paper_id] 标注），流式推送前端

架构说明：原"分析(深度+全局)"与"写作"已合并为这一步——分析产出（领域分析）
经核实只被写作消费、不外传（前端不展示、校验只校验综述），是纯中间产物，故并入
写作，一个 LLM 读证据直接出带标注的综述，省一轮调用且更连贯。聚类保留（算法定方向，
其主题名作为综述章节结构）。原 sub_analyse / sub_writing 整套老代码保留不动（v2 复用）。
"""
import json
import re
from typing import List, Dict, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult

from src.core.prompts import review_writing_prompt
from src.core.model_client import create_subwriting_writing_model_client
from src.core.state_models import ExecutionState, BackToFrontData
from src.agents.sub_analyse_agent.cluster_agent import PaperClusterAgent
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk

logger = setup_logger(__name__)

model_client = create_subwriting_writing_model_client()
MIN_REVIEW_CHARS = 300
_SRC_PATTERN = re.compile(r"\[来源[:：]\s*([^\]]+)\]")


def _build_clusters_input(clusters) -> List[Dict[str, Any]]:
    """把聚类结果组织成"按方向分组的论文证据"，供写作模型顺着方向组织综述。

    保留分簇结构（而非论文平铺）：方向 = 章节蓝本，每条论文带 paper_id 供溯源标注。
    """
    clusters_input = []
    for c in clusters:
        papers = []
        for p in c.papers:
            km = p.get("key_methodology") or {}
            papers.append({
                "paper_id": p.get("paper_id"),
                "核心问题": p.get("core_problem"),
                "方法": km.get("name") if isinstance(km, dict) else None,
                "方法原理": km.get("principle") if isinstance(km, dict) else None,
                "创新点": km.get("novelty") if isinstance(km, dict) else None,
                "主要结果": p.get("main_results"),
                "局限": p.get("limitations"),
            })
        clusters_input.append({
            "研究方向": c.theme_description,
            "关键词": c.keywords,
            "论文数": len(c.papers),
            "论文证据": papers,
        })
    return clusters_input


def _build_prompt(user_request: str, clusters_input: List[Dict[str, Any]], rewrite_feedback: str = "") -> str:
    feedback_block = ""
    if rewrite_feedback:
        feedback_block = f"""
【重写要求】上一稿综述未完全通过忠实度校验，以下论断存在问题，请在重写时修正：
{rewrite_feedback}
修正方式：对不忠实的论断，依据证据如实改写或删除；证据不足以支撑的，写明"现有论文证据不足以支撑"，不要强行圆说或编造。其余忠实的内容可保留。
"""
    return f"""请基于以下按研究方向聚类的论文证据，撰写一篇完整的学术文献综述。

【用户的原始需求】（若其中含篇幅/风格/格式要求，优先遵从）
{user_request}

【按研究方向分组的论文证据（每篇带 paper_id，用于标注来源 [来源: paper_id]）】
{json.dumps(clusters_input, ensure_ascii=False, indent=2)}
{feedback_block}
请一次性写出整篇综述：以上述各研究方向作为章节主线展开，论断句末标注 [来源: paper_id]，证据外内容不得编造。
"""


def _validate_review_output(review_md: str, paper_count: int) -> str | None:
    """写作结果的业务兜底校验，返回失败原因；通过返回 None。"""
    compact = re.sub(r"\s+", "", review_md or "")
    if len(compact) < MIN_REVIEW_CHARS:
        return f"综述正文过短，少于 {MIN_REVIEW_CHARS} 字"

    source_count = len(_SRC_PATTERN.findall(review_md or ""))
    if source_count == 0:
        return "综述缺少来源标注"
    min_sources = min(3, paper_count)
    if source_count < min_sources:
        return f"来源标注过少，当前 {source_count} 个，至少需要 {min_sources} 个"
    return None


async def _generate_review(
    prompt: str,
    state_queue,
    retry_reason: str | None = None,
    stream_to_frontend: bool = True,
) -> str:
    """调用写作模型生成整篇综述。"""
    final_prompt = prompt
    if retry_reason:
        final_prompt = "\n".join(
            [
                prompt,
                "",
                "【必须修正】上一稿未通过系统校验：",
                retry_reason,
                "请重新输出一篇完整综述。正文不少于 300 字；每个关键论断句末必须使用 [来源: paper_id] 标注；不要输出过程说明。",
            ]
        )

    writing_agent = AssistantAgent(
        name="review_writing_agent",
        model_client=model_client,
        system_message=review_writing_prompt,
        model_client_stream=True,
    )

    review_md = ""
    is_thinking = None
    is_first = True
    async for chunk in writing_agent.run_stream(task=final_prompt):
        if is_first:
            is_first = False
            continue
        if isinstance(chunk, TaskResult):
            continue
        if chunk.type == "ThoughtEvent":
            continue
        if chunk.type == "TextMessage":
            review_md = chunk.content
            break
        if not stream_to_frontend:
            continue
        disp, is_thinking = handlerChunk(is_thinking, chunk.content)
        if disp is None:
            continue
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state=disp, data=chunk.content))
    return review_md


async def review_writing_node(state) -> Dict[str, Any]:
    """综述写作节点：聚类定方向 → 一个 LLM 直接写出整篇带标注综述。"""
    state_queue = state["state_queue"]
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.WRITING
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="initializing", data=None))

        # 1. 聚类（算法步，复用现有聚类智能体；其主题名即综述章节结构）
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="thinking", data="正在按研究方向聚类\n"))
        cluster_agent = PaperClusterAgent()
        clusters = await cluster_agent.run(current_state.extracted_data)
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="thinking", data=f"聚类完成，共 {len(clusters)} 个研究方向\n"))

        # 2. 写作（一个 LLM，按方向分组的证据 → 整篇综述，流式）
        clusters_input = _build_clusters_input(clusters)

        # 重写场景：若上一轮忠实度校验判 partial，把不达标论断作为反馈传入，整篇重写
        rewrite_feedback = ""
        fr = current_state.faithfulness_report or {}
        if fr.get("verdict") == "partial" and fr.get("detail"):
            rewrite_feedback = "\n".join(fr["detail"])

        prompt = _build_prompt(current_state.user_request, clusters_input, rewrite_feedback)

        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="thinking", data="正在撰写综述\n"))
        paper_count = len(getattr(current_state.extracted_data, "papers", []) or [])
        review_md = await _generate_review(prompt, state_queue, stream_to_frontend=True)

        invalid_reason = _validate_review_output(review_md, paper_count)
        if invalid_reason:
            await state_queue.put(BackToFrontData(
                step=ExecutionState.WRITING,
                state="thinking",
                data=f"综述初稿未通过结构校验：{invalid_reason}，正在重试一次\n",
            ))
            review_md = await _generate_review(
                prompt,
                state_queue,
                retry_reason=invalid_reason,
                stream_to_frontend=False,
            )
            invalid_reason = _validate_review_output(review_md, paper_count)
            if invalid_reason:
                msg = f"综述生成结果缺少可校验证据标注：{invalid_reason}。请重试或调整召回论文。"
                current_state.error.writing_node_error = msg
                await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="error", data=msg))
                return {"value": current_state}

        current_state.review_markdown = review_md
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="completed", data=review_md))
        return {"value": current_state}

    except Exception as e:
        import traceback
        err_msg = f"Writing failed: {str(e)}"
        logger.error(f"{err_msg}\n{traceback.format_exc()}")
        current_state.error.writing_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING, state="error", data=err_msg))
        return {"value": current_state}
