from src.utils.log_utils import setup_logger
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import asyncio
import json
from pathlib import Path
from src.core.state_models import BackToFrontData
from src.core.config import config
from src.agents.proposal_pipeline.frontend_orchestrator import (
    ProposalFrontendOrchestrator,
    ProposalInputHub,
)

logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
proposal_input_hub = ProposalInputHub()

# CORS（开发期放开；生产请限定域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/send_input")
async def send_input(data: dict):
    """前端回传人工卡点的输入（关键词确认 / 召回论文确认）。"""
    user_input = data.get("input")
    logger.info(f"[/send_input] 收到前端输入: {str(user_input)[:120]}")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})


@app.post("/proposal_input")
async def proposal_input(data: dict):
    """V2 前端回传人工卡点输入。"""
    session_id = data.get("session_id")
    if not session_id:
        return JSONResponse({"status": 400, "msg": "缺少 session_id"}, status_code=400)
    logger.info(
        "[/proposal_input] 收到 V2 卡点输入 session_id=%s type=%s decision=%s keys=%s",
        session_id,
        data.get("type"),
        data.get("decision"),
        sorted(data.keys()),
    )
    await proposal_input_hub.submit(session_id, data)
    return JSONResponse({"status": 200, "msg": "已收到 V2 人工输入"})


@app.get('/api/research')
async def research_stream(query: str):
    """导航型综述生成：SSE 流式推送进度与综述。"""
    from src.agents.review_pipeline.orchestrator import ReviewOrchestrator

    logger.info(f"[/api/research] 新请求 query={query}")
    state_queue = asyncio.Queue()

    async def event_generator():
        while True:
            item = await state_queue.get()
            if item is None:
                logger.info("[SSE] 流程结束，关闭连接")
                yield {"data": '{"step":"","state":"finished","data":null}'}
                break
            # 进度类消息记一行（综述正文流式块太碎，不逐块记）
            if item.state != "generating":
                logger.info(f"[SSE→前端] step={item.step} state={item.state}")
            yield {"data": item.model_dump_json()}

    async def run_and_signal():
        try:
            orchestrator = ReviewOrchestrator(state_queue=state_queue)
            await orchestrator.run(user_request=query)
        except Exception as e:
            logger.error(f"[流程异常] {e}", exc_info=True)
        finally:
            await state_queue.put(None)   # 通知 SSE 结束

    asyncio.create_task(run_and_signal())
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/proposal")
async def proposal_stream(session_id: str, user_request: str, mode: str = "normal", plan: str | None = None):
    """V2 研究方向推荐：SSE 推送完整生成流程。"""
    logger.info(f"[/api/proposal] 新请求 session_id={session_id} mode={mode} user_request={user_request}")
    state_queue = asyncio.Queue()
    proposal_input_hub.ensure(session_id)

    async def event_generator():
        while True:
            item = await state_queue.get()
            if item is None:
                yield {"data": '{"step":"","state":"finished","data":null}'}
                break
            if item.state != "generating":
                logger.info(f"[V2 SSE→前端] step={item.step} state={item.state}")
            yield {"data": item.model_dump_json()}

    async def run_and_signal():
        try:
            orchestrator = ProposalFrontendOrchestrator(
                session_id=session_id,
                user_request=user_request,
                mode=mode,
                raw_plan=plan,
                input_hub=proposal_input_hub,
                state_queue=state_queue,
            )
            await orchestrator.run()
        except Exception as e:
            logger.error(f"[V2 流程异常] {e}", exc_info=True)
            await state_queue.put(BackToFrontData(step="proposal", state="error", data=str(e)))
        finally:
            proposal_input_hub.cleanup(session_id)
            await state_queue.put(None)

    asyncio.create_task(run_and_signal())
    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/proposal/latest_report")
async def latest_proposal_report():
    """读取最近一次 V2 最终报告，不触发新流程。"""
    proposal_root = Path(config.get("SAVE_DIR")) / "proposal_runs"
    report_paths = sorted(
        proposal_root.glob("*/final_report_probe.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not report_paths:
        return JSONResponse({"status": 404, "msg": "没有找到 V2 最终报告"}, status_code=404)

    report_json_path = report_paths[0]
    report_md_path = report_json_path.with_suffix(".md")
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    markdown = report_md_path.read_text(encoding="utf-8") if report_md_path.exists() else ""
    return JSONResponse(
        {
            "status": 200,
            "run_dir": str(report_json_path.parent),
            "report": report,
            "markdown": markdown,
        }
    )


@app.get("/api/research/latest_report")
async def latest_research_report():
    """读取最近一次 V1 综述报告，不触发新流程。"""
    from src.agents.review_pipeline.latest_report import load_latest_review_report

    latest = load_latest_review_report()
    if not latest:
        return JSONResponse({"status": 404, "msg": "没有找到 V1 综述报告"}, status_code=404)
    return JSONResponse({"status": 200, **latest})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
