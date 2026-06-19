from autogen_agentchat.agents import AssistantAgent
# from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional,Dict,Any
from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_default_client, create_reading_model_client
from src.core.state_models import BackToFrontData
from src.core.state_models import State,ExecutionState
from src.services.chroma_client import ChromaClient
from src.knowledge.knowledge import knowledge_base
from src.core.config import config
import json
import asyncio
from src.utils.llm_json import parse_llm_json

logger = setup_logger(__name__)

class KeyMethodology(BaseModel):
    name: Optional[str] = Field(default=None, description="方法名称（如“Transformer-based Sentiment Classifier”）")
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(default=None, description="创新点（如“首次引入领域自适应预训练”）")


class ExtractedPaperData(BaseModel):
    paper_id: Optional[str] = Field(default=None, description="论文ID（代码层回填，关联搜索结果）")
    core_problem: str = Field(default=None, description="核心问题")
    key_methodology: KeyMethodology = Field(default=None, description="关键方法")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")

    @field_validator("core_problem", "main_results", "limitations", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if v is None:
            return ""
        return str(v)

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData] = Field(default=[], description="提取的论文数据列表")

model_client = create_reading_model_client()


def _build_read_agent() -> AssistantAgent:
    """每篇论文用独立的 agent 实例，避免并发抽取时共享对话历史导致串扰。

    AssistantAgent 自带 model context（对话历史），多篇论文并发复用同一个
    全局实例时，第一篇的内容会污染后续所有抽取（实测 47 篇方法名全被带成
    第一篇的 PrML）。model_client 本身无状态，可安全复用。
    """
    return AssistantAgent(
        name="read_agent",
        model_client=model_client,
        system_message=reading_agent_prompt,
    )

def sanitize_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    new_meta = {}
    for k, v in paper.items():
        if v is None:
            continue
        if isinstance(v, list):
            new_meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            new_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            new_meta[k] = v
    return new_meta


async def add_papers_to_kb(papers:Optional[List[Dict[str, Any]]], extracted_papers: ExtractedPapersData):
    """将提取的论文数据添加到知识库"""
    embedding_dic = config.get("embedding-model")
    embedding_provider = embedding_dic.get("model-provider")
    provider_dic = config.get(embedding_provider)
    
    embed_info = {
        "name": embedding_dic.get("model"),
        "dimension": embedding_dic.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }
    kb_type = config.get("KB_TYPE")
    database_info = await knowledge_base.create_database(
        "临时知识库", "用于存储临时提取的论文数据，仅用于本次报告的生成，用完即删", kb_type=kb_type, embed_info=embed_info, llm_info=None,
    )
    db_id = database_info["db_id"]
    config.set("tmp_db_id", db_id) # 记录临时知识库的db_id，后面retrieval_agent中使用
    
    # 注释掉原本的代码，因为papers中包含了一些None值，导致报错
    # documents = [json.dumps(paper.model_dump(), ensure_ascii=False) for paper in extracted_papers.papers],
    # metadatas = [paper for paper in papers],
    # ids = [str(i) for i in range(len(papers))]
    
    documents=[json.dumps(paper.model_dump(),ensure_ascii=False) for paper in extracted_papers.papers]
    sanitized_metadatas = []
    if papers:
        for paper in papers:
           # new_meta = {}
           # for k, v in paper.items():
            #     if isinstance(v, list):
            #         new_meta[k] = ", ".join(str(x) for x in v)
            #     else:
            #         new_meta[k] = v
            # sanitized_metadatas.append(new_meta)
            sanitized_metadatas.append(sanitize_metadata(paper))          
    metadatas = sanitized_metadatas
    
    # # 确保 ids, metadatas 和 documents 长度一致
    # # 注意：这里假设 extracted_papers.papers 和 papers 是一一对应的
    # min_len = min(len(documents), len(metadatas))
    # documents = documents[:min_len]
    # metadatas = metadatas[:min_len]
    # ids = [str(i) for i in range(min_len)]
    ids = [str(i) for i in range(len(documents))] 
    
    data = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    await knowledge_base.add_processed_content(db_id, data)


def _parse_extracted(raw_content, paper_id: str) -> ExtractedPaperData:
    """将 read_agent 的原始输出解析为 ExtractedPaperData，失败则抛异常。

    多级容错：ExtractedPaperData 直通 -> dict -> str(剥 markdown -> json -> ast)
    -> 解列表/papers/paper 包裹 -> model_validate。paper_id 代码层回填，不信任 LLM。
    """
    if isinstance(raw_content, ExtractedPaperData):
        raw_content.paper_id = paper_id
        return raw_content

    data = parse_llm_json(raw_content)

    # 数据结构修正（处理列表包裹或 {"papers": ...} / {"paper": ...} 包裹）
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Parsed content is an empty list")
        data = data[0]
    if isinstance(data, dict):
        if "papers" in data and isinstance(data["papers"], list) and data["papers"]:
            data = data["papers"][0]
        elif "paper" in data and isinstance(data["paper"], dict):
            data = data["paper"]

    parsed_paper = ExtractedPaperData.model_validate(data)
    parsed_paper.paper_id = paper_id  # 代码层回填，不依赖 LLM
    return parsed_paper


async def _read_one_paper(paper: Dict[str, Any]) -> ExtractedPaperData:
    """抽取单篇论文：LLM 抽取 + 解析。失败自动重试 1 次，仍失败则抛异常。

    重试针对 LLM 偶发抽风（输出格式坏、调用异常）——能自愈的自愈，
    真救不了的抛给上层统计为失败篇数，不惊动用户。
    """
    paper_id = paper.get("paper_id")
    read_agent = _build_read_agent()  # 每篇独立实例，避免并发串扰
    last_err = None
    for attempt in range(2):  # 首次 + 重试 1 次
        try:
            result = await read_agent.run(task=str(paper))
            return _parse_extracted(result.messages[-1].content, paper_id)
        except Exception as e:
            last_err = e
            if attempt == 0:
                logger.warning(f"论文 {paper_id} 抽取失败，重试一次。原因: {e}")
    raise RuntimeError(f"论文 {paper_id} 抽取重试后仍失败: {last_err}")


async def reading_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    await state_queue.put(BackToFrontData(step=ExecutionState.READING,state="initializing",data=None))

    papers = current_state.search_results

    try:
        # 并行抽取每篇论文，单篇失败自动重试 1 次。
        # return_exceptions=True：一篇彻底失败不拖垮其余成功的论文。
        results = await asyncio.gather(
            *[_read_one_paper(paper) for paper in papers],
            return_exceptions=True,
        )

        # 收集成功项，统计失败篇数
        extracted_papers = ExtractedPapersData()
        successful_papers = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, ExtractedPaperData):
                extracted_papers.papers.append(result)
                successful_papers.append(papers[i])
            else:
                failed_count += 1  # result 是异常（重试后仍失败）

        # 存入临时向量数据库
        await add_papers_to_kb(successful_papers, extracted_papers)

        current_state.extracted_data = extracted_papers

        # 透明告知真实数量：成功几篇、失败几篇
        success_count = len(extracted_papers.papers)
        if success_count < 3:
            msg = (
                f"成功抽取的论文只有 {success_count} 篇，不足以生成导航型综述。"
                "请扩大关键词、放宽时间范围，或重新确认召回论文。"
            )
            await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="error", data=msg))
            current_state.error.reading_node_error = msg
        elif success_count <= 10:
            warning = f"本次成功抽取 {success_count} 篇论文，证据规模偏小，生成结果更适合作为初步研究地图。"
            current_state.review_warning = warning
            if failed_count > 0:
                await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed",
                    data=f"{warning}（另有 {failed_count} 篇抽取失败已跳过）"))
            else:
                await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed", data=warning))
        elif failed_count > 0:
            await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed",
                data=f"论文阅读完成，成功 {success_count} 篇（{failed_count} 篇抽取失败已跳过）"))
        else:
            await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed",
                data=f"论文阅读完成，共阅读 {success_count} 篇论文"))
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Reading failed: {str(e)}"
        logger.error(err_msg)
        current_state.error.reading_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="error", data=err_msg))
        return {"value": current_state}


if __name__ == "__main__":
    paper = {
        'core_problem': 'Despite the rapid introduction of autonomous vehicles, public misunderstanding and mistrust are prominent issues hindering their acceptance.'
    }
    chroma_client = ChromaClient()
    chroma_client.add_documents(
        documents=[paper],
        metadatas=[paper],
    )   
