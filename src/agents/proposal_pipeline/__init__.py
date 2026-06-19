"""选题建议系统 pipeline。"""

__all__ = ["build_markdown", "build_report", "run_pipeline", "write_report"]


def __getattr__(name: str):
    """惰性导出，避免仅导入报告节点时初始化搜索模型或知识库。"""
    if name == "run_pipeline":
        from src.agents.proposal_pipeline.orchestrator import run_pipeline

        return run_pipeline
    if name in {"build_markdown", "build_report", "write_report"}:
        from src.agents.proposal_pipeline import report_node

        return getattr(report_node, name)
    raise AttributeError(name)
