import os

from src.core.config import config
from .factory import KnowledgeBaseFactory
from .implementations.chroma import ChromaKB
from .manager import KnowledgeBaseManager

# 注册知识库类型
KnowledgeBaseFactory.register("chroma", ChromaKB, {"description": "基于 ChromaDB 的轻量级向量知识库，适合开发和小规模"})


# 创建知识库管理器
work_dir = os.path.join(config.get("SAVE_DIR"), "knowledge_base_data")
knowledge_base = KnowledgeBaseManager(work_dir)


__all__ = ["knowledge_base"]
