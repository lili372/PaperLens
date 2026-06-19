"""综述系统 - 检索节点（含两个人工卡点）

流程：LLM 提关键词 → 【卡点1: 关键词确认】→ 检索 arxiv
      → 【卡点2: 召回论文确认】→ 交下游
两个卡点都复用 userProxyAgent 的 future 暂停机制（on_messages 挂起，前端 /send_input 唤醒）。

与原 search_agent.py 的区别：新增卡点2（召回论文列表确认），让用户在抽取前拦下
"一眼跑偏"的召回，避免浪费下游 token。原 search_node 保留不动（v2 复用）。
"""
import json
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from src.agents.search_agent import search_agent, parse_search_query, FEW_RESULTS_THRESHOLD
from src.agents.userproxy_agent import userProxyAgent
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import ExecutionState, BackToFrontData
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


async def review_search_node(state) -> dict:
    """综述检索节点：关键词确认 + 召回论文确认 两道人工卡点。"""
    state_queue = state["state_queue"]
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="initializing", data=None))

        # 1. LLM 提关键词
        prompt = f"请根据用户查询需求，生成检索查询条件。\n用户查询需求：{current_state.user_request}"
        response = await search_agent.run(task=prompt)
        raw_query = response.messages[-1].content

        # —— 卡点1：关键词确认（推给前端待确认，用户可编辑后回传）——
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="user_review", data=f"{raw_query}"))
        logger.info(f"[卡点1] 已推送关键词待确认，等待前端输入… raw={raw_query[:120]}")
        result = await userProxyAgent.on_messages(
            [TextMessage(content="请确认检索关键词", source="AI")],
            cancellation_token=CancellationToken(),
        )
        logger.info(f"[卡点1] 收到前端确认: {str(result.content)[:120]}")
        search_query = parse_search_query(result.content)

        # 解析失败（四层容错全败 → querys 空）：偶发格式坏，提示重试，不误导用户改词
        if not search_query.querys:
            msg = "关键词解析失败，请重试或换个说法描述需求"
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            current_state.error.search_node_error = msg
            return {"value": current_state}

        # 2. 检索 arxiv
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="thinking", data="正在检索 arXiv…\n"))
        results = await PaperSearcher().search_papers(
            querys=search_query.querys,
            start_date=search_query.start_date,
            end_date=search_query.end_date,
        )

        if len(results) == 0:
            msg = "没有找到相关论文，请尝试其他查询条件"
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            current_state.error.search_node_error = msg
            return {"value": current_state}

        # —— 卡点2：召回论文确认（推论文列表，用户看着对不对，继续/终止）——
        paper_brief = [
            {"paper_id": r.get("paper_id"), "title": r.get("title"),
             "published": r.get("published"), "authors": (r.get("authors") or [])[:3]}
            for r in results
        ]
        review_payload = {
            "user_request": current_state.user_request,   # 卡点也回显用户原问题
            "querys": search_query.querys,
            "count": len(results),
            "papers": paper_brief,
            "few": len(results) < FEW_RESULTS_THRESHOLD,   # 召回偏少提示
        }
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="papers_review",
                                              data=json.dumps(review_payload, ensure_ascii=False)))
        logger.info(f"[卡点2] 已推送 {len(results)} 篇召回论文待确认，等待前端输入…")
        decision = await userProxyAgent.on_messages(
            [TextMessage(content="请确认召回论文是否对路", source="AI")],
            cancellation_token=CancellationToken(),
        )
        logger.info(f"[卡点2] 收到前端决定: {str(decision.content)[:60]}")
        # 用户终止 → 回到改关键词（这里作为节点错误返回，由前端引导退回卡点1重来）
        if str(decision.content).strip().lower() in ("abort", "terminate", "终止", "返回", "no"):
            msg = "已终止：召回论文与主题不符，请返回修改关键词"
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            current_state.error.search_node_error = msg
            return {"value": current_state}

        # 3. 确认通过 → 存结果交下游
        current_state.search_results = results
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="completed",
                                              data=f"检索完成，共 {len(results)} 篇论文"))
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        logger.error(err_msg)
        state["value"].error.search_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=err_msg))
        return {"value": state["value"]}
