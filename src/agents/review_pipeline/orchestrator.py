"""综述系统 - 编排器（LangGraph）

流程：search → reading → review_writing（内含聚类）→ review_faithfulness → END
- 检索用 review_search_node（含关键词确认 + 召回论文确认两道人工卡点）；抽取复用原 reading_node。
- 后半段用 review_pipeline 的新节点：分析已合并进写作（一步成文），校验为整篇版。
- 忠实度修订在校验节点内闭环（judge 给修订句 → 文本替换 → 只重校改动条），
  不再回写作重写。校验出口只剩：pass/partial 放行（partial 带警告），mismatch 终止。

与原 orchestrator.py 的区别：无独立 analyse_node（已并入写作）、无 report_node
（一步成文无需拼接，忠实度警告在 faithfulness 报告里）。原 orchestrator 保留不动（v2 复用）。
"""
from langgraph.graph import StateGraph, END, START

from src.core.state_models import PaperAgentState, ExecutionState, NodeError, State, ConfigSchema, BackToFrontData
from src.agents.review_pipeline.search_node import review_search_node
from src.agents.reading_agent import reading_node
from src.agents.review_pipeline.writing_node import review_writing_node
from src.agents.review_pipeline.faithfulness_node import review_faithfulness_node
from src.agents.review_pipeline.latest_report import save_review_report
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class ReviewOrchestrator:
    """导航型综述编排器。"""

    def __init__(self, state_queue):
        self.state_queue = state_queue
        self.graph = self._build_graph()

    async def handle_error_node(self, state: State) -> State:
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAILED
        logger.error(f"流程失败于 {current_state.current_step}: {current_state.error}")
        return {"value": current_state}

    def condition_handler(self, state: State) -> str:
        """前向路由 + 忠实度校验的回退/分层路由。"""
        current_state = state["value"]
        err = current_state.error
        step = current_state.current_step

        if err.search_node_error is None and step == ExecutionState.SEARCHING:
            return "reading_node"
        elif err.reading_node_error is None and step == ExecutionState.READING:
            return "review_writing_node"   # 抽取后直接进写作（聚类已并入写作节点）
        elif err.writing_node_error is None and step == ExecutionState.WRITING:
            return "review_faithfulness_node"
        elif err.faithfulness_node_error is None and step == ExecutionState.FAITHFULNESS_CHECKING:
            report = current_state.faithfulness_report or {}
            verdict = report.get("verdict", "pass")
            if verdict == "mismatch":
                # 整体造假/不忠实率超阈值 = 检索召回与主题不匹配，根因在上游，终止反馈
                current_state.error.error = (
                    "检索召回的论文与研究主题不匹配，无法生成忠实的综述。"
                    f"（造假率 {report.get('fab_rate')}, 不忠实率 {report.get('unfaithful_rate')}）"
                    "建议收窄或修正检索关键词后重试。"
                )
                return "handle_error_node"
            else:
                # pass 或 partial 都结束：定点修订已在校验节点内闭环完成
                # （partial = 仍有个别改不好的论断、带警告放行）。不再回写作。
                return END
        else:
            return "handle_error_node"

    def _build_graph(self):
        builder = StateGraph(State, context_schema=ConfigSchema)

        builder.add_node("search_node", review_search_node)
        builder.add_node("reading_node", reading_node)
        builder.add_node("review_writing_node", review_writing_node)
        builder.add_node("review_faithfulness_node", review_faithfulness_node)
        builder.add_node("handle_error_node", self.handle_error_node)

        builder.add_edge(START, "search_node")
        builder.add_conditional_edges("search_node", self.condition_handler)
        builder.add_conditional_edges("reading_node", self.condition_handler)
        builder.add_conditional_edges("review_writing_node", self.condition_handler)
        builder.add_conditional_edges("review_faithfulness_node", self.condition_handler)
        builder.add_edge("handle_error_node", END)

        return builder.compile()

    async def run(self, user_request: str, max_papers: int = 50):
        logger.info("综述流程启动...")
        initial_state = PaperAgentState(
            user_request=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config={},
        )
        result = await self.graph.ainvoke({"state_queue": self.state_queue, "value": initial_state})
        final_state = result.get("value") if isinstance(result, dict) else None
        has_error = False
        if final_state and final_state.error:
            has_error = any(value for value in final_state.error.model_dump().values())
        if final_state and not has_error and final_state.review_markdown:
            saved = save_review_report(final_state)
            if saved:
                logger.info(f"V1 最新综述报告已保存: {saved['run_dir']}")
