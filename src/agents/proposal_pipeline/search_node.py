"""选题建议系统 - 检索节点。

职责：把用户研究方向转成"base 论文池 + module 论文池"的检索计划，
经人工确认后检索 arXiv，并输出带角色标记的论文元数据列表。
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import arxiv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from src.agents.userproxy_agent import userProxyAgent
from src.core.config import config
from src.core.model_client import create_search_model_client
from src.core.prompts import proposal_search_agent_prompt
from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    ProposalSearchPlan,
    State,
)
from src.agents.proposal_pipeline.cache_utils import ensure_run_dirs, get_proposal_cache_dir, now_iso, write_json
from src.tasks.paper_search import PaperSearcher
from src.utils.log_utils import setup_logger
from src.utils.llm_json import LLMJsonParseError, parse_llm_json_object

logger = setup_logger(__name__)

FEW_RESULTS_THRESHOLD = config.get_int("search_few_results_threshold", 5)
DEFAULT_LOOKBACK_YEARS = 3
BASE_MAX_RESULTS = 30
MODULE_MAX_RESULTS = 30
PER_QUERY_SELECTED_LIMIT = 3

proposal_search_agent = AssistantAgent(
    name="proposal_search_agent",
    model_client=create_search_model_client(),
    system_message=proposal_search_agent_prompt,
)


def _extract_json_obj(text: Any) -> Dict[str, Any]:
    """从模型或前端输入中稳健解析 JSON 对象。"""
    try:
        return parse_llm_json_object(text)
    except LLMJsonParseError:
        return {}


def _normalize_queries(value: Any) -> List[str]:
    """清洗检索词，去掉空值和重复项。"""
    if isinstance(value, str):
        raw_queries = [value]
    elif isinstance(value, list):
        raw_queries = value
    else:
        raw_queries = []

    queries: List[str] = []
    seen = set()
    for item in raw_queries:
        query = str(item).strip()
        key = query.lower()
        if query and key not in seen:
            queries.append(query)
            seen.add(key)
    return queries


def parse_proposal_search_plan(text: Any) -> ProposalSearchPlan:
    """解析选题建议模式的检索计划。"""
    data = _extract_json_obj(text)
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    start_date = None if start_date in (None, "", "null", "None") else str(start_date)
    end_date = None if end_date in (None, "", "null", "None") else str(end_date)

    return ProposalSearchPlan(
        base_query=(str(data.get("base_query")).strip() if data.get("base_query") else None),
        module_terms=_normalize_queries(data.get("module_terms"))[:3],
        start_date=start_date,
        end_date=end_date,
        rationale=data.get("rationale"),
    )


def apply_default_date_range(plan: ProposalSearchPlan) -> ProposalSearchPlan:
    """用户未指定时间时，默认检索近三年论文。"""
    if plan.start_date:
        return plan

    current_year = datetime.now().year
    plan.start_date = f"{current_year - DEFAULT_LOOKBACK_YEARS}-01-01"
    return plan


def _merge_paper(
    paper_pool: Dict[str, Dict[str, Any]],
    paper: Dict[str, Any],
    role: str,
    matched_query: str,
) -> None:
    """按 paper_id 去重，并记录论文在 V2 搜索中的角色。"""
    paper_id = paper.get("paper_id")
    if not paper_id:
        return

    existing = paper_pool.setdefault(paper_id, dict(paper))
    roles = set(existing.get("roles") or [])
    roles.add(role)
    existing["roles"] = sorted(roles)

    matched_queries = set(existing.get("matched_queries") or [])
    matched_queries.add(matched_query)
    existing["matched_queries"] = sorted(matched_queries)


def select_papers_for_analysis(
    papers: List[Dict[str, Any]],
    plan: ProposalSearchPlan,
    per_query_limit: int = PER_QUERY_SELECTED_LIMIT,
) -> List[Dict[str, Any]]:
    """按检索桶选择进入后续分析的论文。

    规则：base_query 最多取 3 篇；每个 module_term 组合查询最多取 3 篇。
    先按当前搜索排序遍历，跨桶去重，避免某个热门检索词挤占所有名额。
    """
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    buckets = [plan.base_query] if plan.base_query else []
    buckets.extend([f"{plan.base_query} AND {term}" for term in (plan.module_terms or [])[:3]])

    for bucket in buckets:
        count = 0
        for paper in papers:
            paper_id = paper.get("paper_id")
            if not paper_id or paper_id in selected_ids:
                continue
            matched_queries = paper.get("matched_queries") or []
            if bucket not in matched_queries:
                continue
            selected.append(paper)
            selected_ids.add(paper_id)
            count += 1
            if count >= per_query_limit:
                break

    return selected


def _quote_phrase(phrase: str) -> str:
    """转义 arXiv 短语查询中的双引号。"""
    return phrase.replace('"', '\\"')


def _date_filter(start_date: Optional[str], end_date: Optional[str]) -> str:
    """生成 arXiv submittedDate 过滤条件。"""
    searcher = PaperSearcher()
    start = searcher._format_date(start_date) if start_date else "190001010000"
    end = searcher._format_date(end_date) if end_date else datetime.now().strftime("%Y%m%d2359")
    return f"submittedDate:[{start} TO {end}]"


def build_base_raw_query(base_query: str, start_date: Optional[str], end_date: Optional[str]) -> str:
    """构建目标领域本体检索式。"""
    return f'(all:"{_quote_phrase(base_query)}") AND {_date_filter(start_date, end_date)}'


def build_combo_raw_query(
    base_query: str,
    module_term: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> str:
    """构建 base_query AND module_term 的组合检索式。"""
    return (
        f'(all:"{_quote_phrase(base_query)}" AND all:"{_quote_phrase(module_term)}") '
        f'AND {_date_filter(start_date, end_date)}'
    )


async def proposal_search_node(state: State) -> State:
    """选题建议检索节点：生成检索计划、人工确认、检索并输出论文池。"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.SEARCHING
    await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="initializing", data=None))

    try:
        prompt = f"请根据用户研究方向生成选题建议检索计划。\n用户需求：{current_state.user_request}"
        response = await proposal_search_agent.run(task=prompt)
        raw_plan = response.messages[-1].content

        await state_queue.put(BackToFrontData(
            step=ExecutionState.SEARCHING,
            state="user_review",
            data=str(raw_plan),
        ))
        logger.info(f"[选题检索卡点1] 已推送检索计划待确认: {str(raw_plan)[:160]}")

        result = await userProxyAgent.on_messages(
            [TextMessage(content="请确认选题建议检索计划", source="AI")],
            cancellation_token=CancellationToken(),
        )
        plan = apply_default_date_range(parse_proposal_search_plan(result.content))
        current_state.proposal_search_plan = plan

        if not plan.base_query:
            msg = "检索计划解析失败，请重试或换个说法描述研究方向"
            current_state.error.search_node_error = msg
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            return {"value": current_state}

        await state_queue.put(BackToFrontData(
            step=ExecutionState.SEARCHING,
            state="thinking",
            data="正在按领域本体和模块组合检索 arXiv...\n",
        ))

        searcher = PaperSearcher()
        paper_pool: Dict[str, Dict[str, Any]] = {}

        base_raw_query = build_base_raw_query(plan.base_query, plan.start_date, plan.end_date)
        base_results = await searcher.search_raw_query(
            raw_query=base_raw_query,
            max_results=BASE_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        for paper in base_results:
            _merge_paper(paper_pool, paper, "base", plan.base_query)

        for module_term in plan.module_terms:
            combo_raw_query = build_combo_raw_query(
                base_query=plan.base_query,
                module_term=module_term,
                start_date=plan.start_date,
                end_date=plan.end_date,
            )
            module_results = await searcher.search_raw_query(
                raw_query=combo_raw_query,
                max_results=MODULE_MAX_RESULTS,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )
            for paper in module_results:
                _merge_paper(paper_pool, paper, "module", f"{plan.base_query} AND {module_term}")

        results = list(paper_pool.values())
        results.sort(key=lambda p: (p.get("published_date") or ""), reverse=True)

        if not results:
            msg = "没有找到相关论文，请修改检索计划或放宽时间范围"
            current_state.error.search_node_error = msg
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            return {"value": current_state}

        review_payload = {
            "user_request": current_state.user_request,
            "search_plan": plan.model_dump(),
            "count": len(results),
            "few": len(results) < FEW_RESULTS_THRESHOLD,
            "papers": [
                {
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "published": paper.get("published"),
                    "authors": (paper.get("authors") or [])[:3],
                    "primary_category": paper.get("primary_category"),
                    "categories": paper.get("categories"),
                    "roles": paper.get("roles"),
                    "matched_queries": paper.get("matched_queries"),
                    "summary": (paper.get("summary") or "")[:500],
                    "pdf_url": paper.get("pdf_url"),
                }
                for paper in results
            ],
        }
        await state_queue.put(BackToFrontData(
            step=ExecutionState.SEARCHING,
            state="papers_review",
            data=json.dumps(review_payload, ensure_ascii=False),
        ))
        logger.info(f"[选题检索卡点2] 已推送 {len(results)} 篇论文待确认")

        decision = await userProxyAgent.on_messages(
            [TextMessage(content="请确认召回论文是否适合继续做选题建议", source="AI")],
            cancellation_token=CancellationToken(),
        )
        if str(decision.content).strip().lower() in ("abort", "terminate", "终止", "返回", "no"):
            msg = "已终止：召回论文不适合继续生成选题建议，请返回修改检索计划"
            current_state.error.search_node_error = msg
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=msg))
            return {"value": current_state}

        current_state.search_results = results
        run_dir = get_proposal_cache_dir(plan)
        paths = ensure_run_dirs(run_dir)
        current_state.proposal_run_dir = str(run_dir)
        write_json(paths["run_dir"] / "search_results_all.json", {
            "created_at": now_iso(),
            "user_request": current_state.user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(results),
            "papers": results,
        })
        write_json(paths["search_results"], {
            "created_at": now_iso(),
            "user_request": current_state.user_request,
            "search_plan": plan.model_dump(),
            "raw_count": len(results),
            "selected_count": len(results),
            "selection_policy": {
                "type": "raw_dedup_before_pdf_backfill",
                "note": "PDF 下载节点会按检索桶补位，并覆盖本文件为最终进入分析的成功 PDF 列表。",
                "buckets": [plan.base_query] + [f"{plan.base_query} AND {term}" for term in plan.module_terms],
            },
            "count": len(results),
            "papers": results,
        })
        await state_queue.put(BackToFrontData(
            step=ExecutionState.SEARCHING,
            state="completed",
            data=(
                f"选题建议检索完成，原始去重 {len(results)} 篇，"
                f"等待 PDF 下载节点按桶补位筛选，已缓存到 {paths['search_results']}"
            ),
        ))
        return {"value": current_state}

    except Exception as exc:
        err_msg = f"选题建议检索失败: {str(exc)}"
        logger.error(err_msg)
        current_state.error.search_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING, state="error", data=err_msg))
        return {"value": current_state}
