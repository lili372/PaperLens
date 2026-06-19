"""选题建议系统 - 前端交互编排器。

职责：服务 V2 前端完整流程，处理检索计划确认、PDF 后论文范围确认、
正式流水线执行和最终报告返回。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.proposal_pipeline.cache_utils import ensure_run_dirs, get_proposal_cache_dir
from src.agents.proposal_pipeline.orchestrator import (
    run_pdf_download,
    run_profiles,
    run_rag_build,
    run_report,
    run_review_batch,
    run_revisions,
    run_search,
    run_topics,
    validate_candidate_count,
    validate_profile_count,
)
from src.agents.proposal_pipeline.search_node import (
    apply_default_date_range,
    parse_proposal_search_plan,
    proposal_search_agent,
)
from src.core.state_models import BackToFrontData, ProposalSearchPlan
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class ProposalInputHub:
    """按 session_id 管理 V2 前端人工卡点输入。"""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def ensure(self, session_id: str) -> asyncio.Queue:
        """获取或创建会话队列。"""
        return self._queues.setdefault(session_id, asyncio.Queue())

    async def submit(self, session_id: str, payload: dict[str, Any]) -> None:
        """提交一次前端人工输入。"""
        await self.ensure(session_id).put(payload)

    async def wait(self, session_id: str) -> dict[str, Any]:
        """等待指定会话的下一次人工输入。"""
        return await self.ensure(session_id).get()

    def cleanup(self, session_id: str) -> None:
        """清理会话队列。"""
        self._queues.pop(session_id, None)


class ProposalFrontendOrchestrator:
    """V2 前端专用编排器。"""

    def __init__(
        self,
        *,
        session_id: str,
        user_request: str,
        mode: str,
        raw_plan: str | None,
        input_hub: ProposalInputHub,
        state_queue: asyncio.Queue,
    ) -> None:
        self.session_id = session_id
        self.user_request = user_request
        self.mode = mode
        self.raw_plan = raw_plan
        self.input_hub = input_hub
        self.state_queue = state_queue

    async def emit(self, step: str, state: str, data: Any = None) -> None:
        """向前端推送一个 SSE 状态。"""
        logger.info(
            "[V2 前端编排] session_id=%s emit step=%s state=%s data=%s",
            self.session_id,
            step,
            state,
            str(data)[:300],
        )
        await self.state_queue.put(BackToFrontData(step=step, state=state, data=data))

    def parse_front_plan(self, raw: str | dict[str, Any] | None) -> ProposalSearchPlan:
        """解析前端传入的检索计划。"""
        if raw is None:
            return ProposalSearchPlan()
        if isinstance(raw, dict):
            raw_text = json.dumps(raw, ensure_ascii=False)
        else:
            raw_text = raw
        return apply_default_date_range(parse_proposal_search_plan(raw_text))

    async def build_plan(self) -> ProposalSearchPlan:
        """生成或读取检索计划，并在普通模式下等待用户确认。"""
        logger.info(
            "[V2 前端编排] session_id=%s build_plan mode=%s user_request=%s",
            self.session_id,
            self.mode,
            self.user_request,
        )
        if self.mode == "expert":
            plan = self.parse_front_plan(self.raw_plan)
            if not plan.base_query:
                raise ValueError("专家模式缺少领域本体词 base_query")
            logger.info(
                "[V2 前端编排] session_id=%s expert_plan=%s",
                self.session_id,
                plan.model_dump(),
            )
            return plan

        await self.emit("plan", "initializing", "正在根据研究需求生成检索计划")
        response = await proposal_search_agent.run(
            task=f"请根据用户研究方向生成选题建议检索计划。\n用户需求：{self.user_request}"
        )
        plan = apply_default_date_range(parse_proposal_search_plan(response.messages[-1].content))
        logger.info(
            "[V2 前端编排] session_id=%s generated_plan=%s raw=%s",
            self.session_id,
            plan.model_dump(),
            str(response.messages[-1].content)[:500],
        )
        await self.emit("plan", "plan_review", plan.model_dump())

        front_input = await self.input_hub.wait(self.session_id)
        logger.info(
            "[V2 前端编排] session_id=%s confirmed_plan_input_keys=%s",
            self.session_id,
            sorted(front_input.keys()),
        )
        confirmed = self.parse_front_plan(front_input.get("plan"))
        if not confirmed.base_query:
            raise ValueError("检索计划缺少领域本体词 base_query")
        logger.info(
            "[V2 前端编排] session_id=%s confirmed_plan=%s",
            self.session_id,
            confirmed.model_dump(),
        )
        return confirmed

    @staticmethod
    def build_paper_review_payload(search_all_path: Path, pdf_manifest_path: Path) -> dict[str, Any]:
        """构建 PDF 后论文范围确认数据。"""
        search_all = json.loads(search_all_path.read_text(encoding="utf-8")) if search_all_path.exists() else {}
        pdf_manifest = json.loads(pdf_manifest_path.read_text(encoding="utf-8"))
        raw_count = search_all.get("raw_count") or pdf_manifest.get("attempted_count") or 0
        selected_count = pdf_manifest.get("success_count", 0)
        target_count = (pdf_manifest.get("selection_policy") or {}).get("target_success_count") or selected_count
        not_selected_count = max(0, raw_count - selected_count)
        return {
            "raw_count": raw_count,
            "target_success_count": target_count,
            "pdf_success_count": selected_count,
            "pdf_failed_count": pdf_manifest.get("failed_count", 0),
            "pdf_skipped_count": pdf_manifest.get("skipped_count", 0),
            "not_selected_count": not_selected_count,
            "fill_report": pdf_manifest.get("fill_report") or {},
            "papers": [
                {
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "published": paper.get("published"),
                    "primary_category": paper.get("primary_category"),
                    "matched_queries": paper.get("matched_queries"),
                    "summary": (paper.get("summary") or "")[:360],
                }
                for paper in (pdf_manifest.get("papers") or [])
            ],
        }

    async def wait_paper_confirmation(self) -> str:
        """等待用户确认 PDF 后最终进入分析的论文集合。"""
        decision = await self.input_hub.wait(self.session_id)
        logger.info(
            "[V2 前端编排] session_id=%s paper_confirmation=%s",
            self.session_id,
            decision,
        )
        decision_text = str(decision.get("decision", "")).lower()
        if decision_text in {"continue", "retry_failed_pdfs"}:
            return decision_text
        else:
            raise ValueError("已返回修改检索计划，未继续生成")

    async def run(self) -> None:
        """执行 V2 前端完整流程。"""
        search_plan = await self.build_plan()
        run_dir = get_proposal_cache_dir(search_plan)
        paths = ensure_run_dirs(run_dir)
        logger.info(
            "[V2 前端编排] session_id=%s run_dir=%s paths=%s",
            self.session_id,
            run_dir,
            {key: str(value) for key, value in paths.items()},
        )
        profile_dir = run_dir / "profiles"
        candidates_path = run_dir / "topic_proposals_probe_json.json"
        rag_manifest_path = run_dir / "rag_manifest.json"
        review_dir = run_dir / "candidate_review_batch_claim_buckets"
        revision_dir = run_dir / "revision_batch_probe"
        report_md_path = run_dir / "final_report_probe.md"
        report_json_path = run_dir / "final_report_probe.json"

        await self.emit("search", "thinking", "正在检索 arXiv")
        search_mode = await run_search(search_plan, self.user_request, paths["search_results"], force=False)
        search_payload = json.loads(paths["search_results"].read_text(encoding="utf-8"))
        logger.info(
            "[V2 前端编排] session_id=%s search_mode=%s raw_count=%s count=%s output=%s",
            self.session_id,
            search_mode,
            search_payload.get("raw_count"),
            search_payload.get("count"),
            paths["search_results"],
        )
        await self.emit("search", "completed", f"arXiv 检索完成：{search_mode}")

        pdf_force = False
        while True:
            await self.emit(
                "pdf_download",
                "thinking",
                "正在重新下载失败 PDF" if pdf_force else "正在下载 PDF，并按检索桶补位",
            )
            pdf_mode = await run_pdf_download(
                search_plan,
                self.user_request,
                paths["search_results"],
                paths["pdf_manifest"],
                paths["pdf_dir"],
                force=pdf_force,
            )
            pdf_payload = json.loads(paths["pdf_manifest"].read_text(encoding="utf-8"))
            logger.info(
                "[V2 前端编排] session_id=%s pdf_mode=%s force=%s success=%s failed=%s attempted=%s output=%s",
                self.session_id,
                pdf_mode,
                pdf_force,
                pdf_payload.get("success_count"),
                pdf_payload.get("failed_count"),
                pdf_payload.get("attempted_count"),
                paths["pdf_manifest"],
            )
            await self.emit("pdf_download", "completed", f"PDF 下载完成：{pdf_mode}")

            paper_payload = self.build_paper_review_payload(
                paths["search_results"].with_name("search_results_all.json"),
                paths["pdf_manifest"],
            )
            await self.emit("papers", "paper_review", paper_payload)
            logger.info(
                "[V2 前端编排] session_id=%s paper_review raw=%s success=%s failed=%s paper_ids=%s",
                self.session_id,
                paper_payload.get("raw_count"),
                paper_payload.get("pdf_success_count"),
                paper_payload.get("pdf_failed_count"),
                [paper.get("paper_id") for paper in paper_payload.get("papers", [])],
            )
            paper_decision = await self.wait_paper_confirmation()
            if paper_decision == "continue":
                break
            pdf_force = True

        await self.emit("profiles", "thinking", "正在为已下载论文生成画像")
        profile_mode = await run_profiles(paths["pdf_manifest"], profile_dir, force=False, max_papers=None)
        profile_warning = validate_profile_count(profile_dir)
        logger.info(
            "[V2 前端编排] session_id=%s profile_mode=%s profile_count=%s dir=%s",
            self.session_id,
            profile_mode,
            len(list(profile_dir.glob("paper_profile_*.md"))),
            profile_dir,
        )
        profile_msg = f"论文画像完成：{profile_mode}"
        if profile_warning:
            profile_msg = f"{profile_msg}。{profile_warning}"
        await self.emit("profiles", "completed", profile_msg)

        await self.emit("topic", "thinking", "正在生成候选研究方向")
        topic_mode = await run_topics(profile_dir, candidates_path, force=True)
        topic_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        topic_warning = validate_candidate_count(candidates_path)
        logger.info(
            "[V2 前端编排] session_id=%s topic_mode=%s candidate_count=%s output=%s",
            self.session_id,
            topic_mode,
            len(topic_payload) if isinstance(topic_payload, list) else "unknown",
            candidates_path,
        )
        topic_msg = f"候选方向生成完成：{topic_mode}"
        if topic_warning:
            topic_msg = f"{topic_msg}。{topic_warning}"
        await self.emit("topic", "completed", topic_msg)

        await self.emit("rag", "thinking", "正在解析 PDF 并建立证据库")
        rag_mode, db_id = await run_rag_build(paths["pdf_manifest"], rag_manifest_path, force=False)
        rag_payload = json.loads(rag_manifest_path.read_text(encoding="utf-8"))
        logger.info(
            "[V2 前端编排] session_id=%s rag_mode=%s db_id=%s paper_count=%s chunk_count=%s output=%s",
            self.session_id,
            rag_mode,
            db_id,
            rag_payload.get("paper_count"),
            rag_payload.get("chunk_count"),
            rag_manifest_path,
        )
        await self.emit("rag", "completed", f"PDF 证据库完成：{rag_mode}")

        await self.emit("review_batch", "thinking", "正在批量查证候选方向并审查可做性")
        review_mode = await run_review_batch(
            db_id,
            paths["pdf_manifest"],
            candidates_path,
            profile_dir,
            review_dir,
            force=True,
        )
        review_summary_path = review_dir / "summary.json"
        review_payload = json.loads(review_summary_path.read_text(encoding="utf-8"))
        logger.info(
            "[V2 前端编排] session_id=%s review_mode=%s item_count=%s decisions=%s output=%s",
            self.session_id,
            review_mode,
            len(review_payload.get("items") or []),
            [
                {
                    "candidate_id": item.get("candidate_id"),
                    "decision": item.get("review_decision"),
                }
                for item in (review_payload.get("items") or [])
            ],
            review_summary_path,
        )
        await self.emit("review_batch", "completed", f"查证与审查完成：{review_mode}")

        await self.emit("revision_batch", "thinking", "正在对 revise_keep 方向做单次修改")
        revision_mode = await run_revisions(candidates_path, review_dir, revision_dir, force=True)
        final_candidates_path = revision_dir / "final_candidates_probe.json"
        final_payload = json.loads(final_candidates_path.read_text(encoding="utf-8"))
        logger.info(
            "[V2 前端编排] session_id=%s revision_mode=%s final=%s deferred=%s rejected=%s output=%s",
            self.session_id,
            revision_mode,
            final_payload.get("final_candidate_count"),
            final_payload.get("deferred_candidate_count"),
            final_payload.get("rejected_candidate_count"),
            final_candidates_path,
        )
        await self.emit("revision_batch", "completed", f"修改与分流完成：{revision_mode}")

        await self.emit("report", "thinking", "正在拼接最终中文报告")
        report_mode = run_report(
            paths["search_results"],
            paths["pdf_manifest"],
            revision_dir,
            review_dir,
            report_md_path,
            report_json_path,
            force=True,
        )
        logger.info(
            "[V2 前端编排] session_id=%s report_mode=%s md=%s json=%s",
            self.session_id,
            report_mode,
            report_md_path,
            report_json_path,
        )
        await self.emit("report", "completed", f"最终报告完成：{report_mode}")

        self.write_frontend_manifest(run_dir, search_plan, report_md_path, report_json_path)
        await self.emit(
            "proposal",
            "result",
            {
                "report": json.loads(report_json_path.read_text(encoding="utf-8")),
                "markdown": report_md_path.read_text(encoding="utf-8"),
            },
        )
        logger.info("[V2 前端编排] session_id=%s finished", self.session_id)

    def write_frontend_manifest(
        self,
        run_dir: Path,
        search_plan: ProposalSearchPlan,
        report_md_path: Path,
        report_json_path: Path,
    ) -> None:
        """写入前端触发运行的轻量 manifest。"""
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "frontend",
            "session_id": self.session_id,
            "run_dir": str(run_dir),
            "search_plan": search_plan.model_dump(),
            "outputs": {
                "final_report_md": str(report_md_path),
                "final_report_json": str(report_json_path),
            },
        }
        (run_dir / "pipeline_manifest_frontend.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "[V2 前端编排] session_id=%s frontend_manifest=%s",
            self.session_id,
            run_dir / "pipeline_manifest_frontend.json",
        )
