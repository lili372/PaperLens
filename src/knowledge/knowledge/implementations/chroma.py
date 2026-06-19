import asyncio
import os
import traceback
from typing import Any, Optional, Union, List
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import (
    Embedding,
    PyEmbedding,
    OneOrMany,
)
from src.knowledge.knowledge.base import KnowledgeBase
from src.knowledge.knowledge.indexing import process_file_to_markdown, process_file_to_json, process_url_to_markdown
from src.knowledge.knowledge.utils.kb_utils import (
    get_embedding_config,
    prepare_item_metadata,
    split_text_into_chunks,
    split_text_into_qa_chunks,
    validate_img_embedding_file,
)
from src.utils.datetime_utils import utc_isoformat
from src.utils.log_utils import setup_logger
from src.core.config import config

logger = setup_logger(__name__)


class BatchedEmbeddingFunction:
    """对底层 embedding 函数做分批包装。

    Qwen(DashScope)的 text-embedding 接口单次 batch 上限为 10 条，
    超过会报 400 InvalidParameter。这里把输入切成 <=10 的批逐批调用再拼回，
    对上层（ChromaDB）透明。
    """

    def __init__(self, inner, batch_size: int = 10):
        self._inner = inner
        self._batch_size = batch_size

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        results = []
        for i in range(0, len(input), self._batch_size):
            batch = input[i:i + self._batch_size]
            results.extend(self._inner(batch))
        return results

    # ChromaDB 部分版本会调用这些方法做校验/持久化，转发给底层
    def name(self):
        fn = getattr(self._inner, "name", None)
        return fn() if callable(fn) else "batched_embedding"

    def __getattr__(self, item):
        return getattr(self._inner, item)


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.query

class ChromaKB(KnowledgeBase):
    """基于 ChromaDB 的向量知识库实现"""

    def __init__(self, work_dir: str, **kwargs):
        """
        初始化 ChromaDB 知识库

        Args:
            work_dir: 工作目录
            **kwargs: 其他配置参数
        """
        super().__init__(work_dir)

        if chromadb is None:
            raise ImportError("chromadb is not installed. Please install it with: pip install chromadb")

        # ChromaDB 配置
        self.chroma_db_path = os.path.join(work_dir, "chromadb")
        os.makedirs(self.chroma_db_path, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 存储集合映射 {db_id: collection}
        self.collections: dict[str, Any] = {}
        logger.info("ChromaKB initialized")

    @property
    def kb_type(self) -> str:
        """知识库类型标识"""
        return "chroma"

    async def _create_kb_instance(self, db_id: str, kb_config: dict) -> Any:
        """创建向量数据库集合"""
        logger.info(f"Creating ChromaDB collection for {db_id}")

        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        # embed_info = self.databases_meta[db_id].get("embed_info", {})
        # 先获取原始值（可能是 None、具体值或不存在）
        embed_info = self.databases_meta[db_id].get("embed_info")
        # 确保最终值为 {}（如果是 None 或不存在）
        embed_info = embed_info if embed_info is not None else {}

        embedding_function = self._get_embedding_function(embed_info)

        # 创建或获取集合
        collection_name = db_id

        try:
            # 尝试获取现有集合
            collection = self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
            logger.info(f"Retrieved existing collection: {collection_name}")


        except Exception:
            # 创建新集合
            logger.info(f"Creating new collection with embedding model: {embed_info.get('name', 'default')}")
            collection_metadata = {
                "db_id": db_id,
                "created_at": utc_isoformat(),
                "embedding_model": embed_info.get("name") if embed_info else "default",
            }
            collection = self.chroma_client.create_collection(
                name=collection_name, embedding_function=embedding_function, metadata=collection_metadata
            )
            logger.info(f"Created new collection: {collection_name}")

        return collection

    async def _initialize_kb_instance(self, instance: Any) -> None:
        """初始化向量数据库集合（无需特殊初始化）"""
        pass

    def _get_embedding_function(self, embed_info: dict):
        """获取 embedding 函数（带分批，避免 Qwen 单次 >10 条报 400）"""
        config_dict = get_embedding_config(embed_info)

        inner = OpenAIEmbeddingFunction(
            model_name=config_dict["model"],
            api_key=config_dict["api_key"],
            api_base=config_dict["base_url"].replace("/embeddings", ""),
        )
        return BatchedEmbeddingFunction(inner, batch_size=10)

    async def _get_chroma_collection(self, db_id: str):
        """获取或创建 ChromaDB 集合"""
        if db_id in self.collections:
            return self.collections[db_id]

        if db_id not in self.databases_meta:
            logger.info(f"self.databases_meta中没有{db_id}")
            return None

        try:
            # 创建集合
            collection = await self._create_kb_instance(db_id, {})
            await self._initialize_kb_instance(collection)

            self.collections[db_id] = collection
            return collection

        except Exception as e:
            logger.error(f"Failed to create vector collection for {db_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _get_image_chroma_collection(self, db_id: str):
        """获取或创建图片专用的 ChromaDB 集合（使用512维嵌入）"""
        if db_id not in self.databases_meta:
            return None

        # 为图片嵌入创建专门的集合名称
        image_collection_name = f"{db_id}_images"

        try:
            # 检查集合是否已存在
            if image_collection_name in self.collections:
                return self.collections[image_collection_name]

            # 尝试获取现有集合
            try:
                collection = self.chroma_client.get_collection(name=image_collection_name)
                logger.info(f"Retrieved existing image collection: {image_collection_name}")
                self.collections[image_collection_name] = collection
                return collection
            except Exception:
                # 创建新集合 - 使用自定义嵌入函数，固定维度为512
                # 对于图片嵌入，我们不需要实际的嵌入函数，因为嵌入已经由CLIP模型生成
                # 我们创建一个空的嵌入函数，但指定维度为512
                class ImageEmbeddingFunction:
                    def __init__(self):
                        pass
                    
                    def __call__(self, texts):
                        # 返回与文本数量相同的512维零向量
                        # 实际嵌入会在外部生成
                        return [[0.0] * 512 for _ in texts]

                embed_function = ImageEmbeddingFunction()

                # 创建集合元数据
                collection_metadata = {
                    "db_id": db_id,
                    "created_at": utc_isoformat(),
                    "embedding_model": "clip_image_embedding",
                    "embedding_dimension": 512
                }

                collection = self.chroma_client.create_collection(
                    name=image_collection_name, 
                    embedding_function=embed_function,
                    metadata=collection_metadata
                )
                
                logger.info(f"Created new image collection: {image_collection_name}")
                self.collections[image_collection_name] = collection
                return collection

        except Exception as e:
            logger.error(f"Failed to get/create image collection {image_collection_name}: {e}")
            return None
    def parse_json_into_embedding_chunks(self, json_content: str, file_id: str, filename: str, params: dict) -> list[dict]:
        """将JSON解析成嵌入块"""
        pass
        # import json
        # artifacts = json.loads(json_content)
        # chunks = []
        # for chunk_index, artifact in enumerate(artifacts):
        #     image_url = artifact ["image_url"]
        #     img_desc = get_image_description(image_url)
        #     desc_embedding = get_text_embedding(img_desc)
        #     image_embedding = get_image_embedding(image_url)
        #     img_chunk = {
        #         "content": f"文物名称：{artifact ['name']}\n 对应的文物描述：{artifact ['description']}\n 对应的文物图片URL：{artifact ['image_url']}\n 对应的文物图片的描述：{img_desc}",
        #         "embeddings": image_embedding,
        #         "id": f"{file_id}_chunk_{chunk_index}_img_chunk",
        #         "file_id": file_id,
        #         "filename": filename,
        #         "chunk_index": chunk_index,
        #         "source": filename,
        #         "chunk_id": f"{file_id}_chunk_{chunk_index}",
        #         "metadata": {
        #             "description": artifact ["description"],
        #             "name": artifact ["name"],
        #             "image_url": artifact ["image_url"],
        #             "detail_url": artifact ["detail_url"], 
        #             "full_doc_id": file_id,
        #             "source": filename,
        #             "chunk_id": f"{file_id}_artifact_chunk_{chunk_index}",
        #             "chunk_type": "img_chunk",
        #         }
        #     }
        #     desc_chunk = {
        #         "content": f"文物名称：{artifact ['name']}\n 对应的文物描述：{artifact ['description']}\n 对应的文物图片URL：{artifact ['image_url']}\n 对应的文物图片的描述：{img_desc}",
        #         "embeddings": desc_embedding,
        #         "id": f"{file_id}_chunk_{chunk_index}_desc_chunk",
        #         "file_id": file_id,
        #         "filename": filename,
        #         "chunk_index": chunk_index,
        #         "source": filename,
        #         "chunk_id": f"{file_id}_chunk_{chunk_index}",
        #         "metadata": {
        #             "description": artifact ["description"],
        #             "name": artifact ["name"],
        #             "image_url": artifact ["image_url"],
        #             "detail_url": artifact ["detail_url"], 
        #             "full_doc_id": file_id,
        #             "source": filename,
        #             "chunk_id": f"{file_id}_artifact_chunk_{chunk_index}",
        #             "chunk_type": "desc_chunk",
        #         }
        #     }
        #     chunks.append (img_chunk)
        #     chunks.append (desc_chunk)
        # return chunks

    def split_json_into_chunks(self, json_content: str, file_id: str, filename: str, params: dict) -> list[dict]:
        """将JSON分割成块"""
        pass
        # import json
        # artifacts = json.loads(json_content)
        # chunks = []
        # for chunk_index, artifact in enumerate(artifacts):
        #     content = f"文物名称：{artifact ['name']}\n 文物描述：{artifact ['description']}"
        #     chunk = {
        #         "content": content.strip(),
        #         "id": f"{file_id}_chunk_{chunk_index}",
        #         "file_id": file_id,
        #         "filename": filename,
        #         "chunk_index": chunk_index,
        #         "source": filename,
        #         "chunk_id": f"{file_id}_chunk_{chunk_index}",
        #         "metadata": {
        #             "image_url": artifact ["image_url"],
        #             "detail_url": artifact ["detail_url"], 
        #             "name": artifact ["name"],
        #             "full_doc_id": file_id,
        #             "source": filename,
        #             "chunk_id": f"{file_id}_artifact_chunk_{chunk_index}",
        #             "chunk_type": "normal",
        #         }
        #     }
        #     chunks.append (chunk)
        # return chunks

    def _split_text_into_chunks(self, text: str, file_id: str, filename: str, params: dict) -> list[dict]:
        """将文本分割成块"""
        # 检查是否使用QA分割模式
        use_qa_split = params.get("use_qa_split", False)

        if use_qa_split:
            # 使用QA分割模式
            qa_separator = params.get("qa_separator", "\n\n\n")
            chunks = split_text_into_qa_chunks(text, file_id, filename, qa_separator, params)
        else:
            # 使用传统分割模式
            chunks = split_text_into_chunks(text, file_id, filename, params)

        # 为 ChromaDB 添加特定的 metadata 格式
        for chunk in chunks:
            chunk["metadata"] = {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "full_doc_id": file_id,
                "chunk_type": chunk.get("chunk_type", "normal"),  # 添加chunk类型标识
            }

        return chunks

    async def add_processed_content(self, db_id: str, data: dict | None = None) -> list[dict]:
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Failed to get ChromaDB collection for {db_id}")
        try:
            batch_embeddings = data.get("embeddings", None)
            batch_documents = data.get("documents", [])
            batch_metadatas = data.get("metadatas", None)
            batch_ids = data.get("ids", None)

            await asyncio.to_thread(
                                collection.add,
                                embeddings=batch_embeddings,
                                documents=batch_documents,
                                metadatas=batch_metadatas,
                                ids=batch_ids,
                            )
            logger.info(f"成功将 {len(batch_ids)} 个项 插入到临时知识库中")
        except Exception as e:
            logger.error(f"错误处理{len(batch_ids)} 个项 插入到临时知识库中，失败: {e}, {traceback.format_exc()}")

    async def add_content(self, db_id: str, items: list[str], params: dict | None) -> list[dict]:
        """添加内容（文件/URL）"""
        if db_id not in self.databases_meta:
            raise ValueError(f"Database {db_id} not found")

        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Failed to get ChromaDB collection for {db_id}")

        content_type = params.get("content_type", "file") if params else "file"
        processed_items_info = []

        for item in items:
            # 准备文件元数据
            metadata = prepare_item_metadata(item, content_type, db_id)
            file_id = metadata["file_id"]
            filename = metadata["filename"]

            # 添加文件记录
            file_record = metadata.copy()
            self.files_meta[file_id] = file_record
            self._save_metadata()

            self._add_to_processing_queue(file_id)

            file_path_obj = Path(item)
            file_ext = file_path_obj.suffix.lower()

            try:
                # 根据内容类型处理内容
                if content_type == "file":
                    markdown_content = await process_file_to_markdown(item, params=params)
                else:  # URL    
                    markdown_content = await process_url_to_markdown(item, params=params)
            
                
                chunks = []
                chunks = self._split_text_into_chunks(markdown_content, file_id, filename, params)
                logger.info(f"Split {filename} into {len(chunks)} chunks")

                # 准备向量数据库插入的数据
                if chunks:
                    documents = [chunk["content"] for chunk in chunks]
                    metadatas = [chunk["metadata"] for chunk in chunks]
                    ids = [chunk["id"] for chunk in chunks]

                    # 插入到 ChromaDB - 分批处理以避免超出 OpenAI 批次大小限制
                    batch_size = 64  # OpenAI 的最大批次大小限制
                    total_batches = (len(chunks) + batch_size - 1) // batch_size

                    for i in range(0, len(chunks), batch_size):
                        batch_documents = documents[i : i + batch_size]
                        batch_metadatas = metadatas[i : i + batch_size]
                        batch_ids = ids[i : i + batch_size]

                        await asyncio.to_thread(
                            collection.add,
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            ids=batch_ids,
                        )

                        batch_num = i // batch_size + 1
                        logger.info(f"Processed batch {batch_num}/{total_batches} for {filename}")

                logger.info(f"Inserted {content_type} {item} into ChromaDB. Done.")

                # 更新状态为完成
                self.files_meta[file_id]["status"] = "done"
                self._save_metadata()
                file_record["status"] = "done"

            except Exception as e:
                logger.error(f"处理{content_type} {item} 失败: {e}, {traceback.format_exc()}")
                self.files_meta[file_id]["status"] = "failed"
                self._save_metadata()
                file_record["status"] = "failed"
            finally:
                self._remove_from_processing_queue(file_id)

            processed_items_info.append(file_record)

        return processed_items_info

    
    async def add_image_embeddings(self, db_id: str, item: str, params: dict | None):
        """添加图片嵌入"""
        pass
        # 校验格式
        # if not validate_img_embedding_file(item):
        #     return
        # if db_id not in self.databases_meta:
        #     raise ValueError(f"Database {db_id} not found")

        # collection = await self._get_image_chroma_collection(db_id)
        # if not collection:
        #     raise ValueError(f"Failed to get ChromaDB collection for {db_id}")

        # content_type = params.get("content_type", "file") if params else "file"
        # # processed_items_info = []

        # # 准备文件元数据
        # metadata = prepare_item_metadata(item, content_type, db_id)
        # file_id = metadata["file_id"]
        # filename = metadata["filename"]

        # 添加文件记录
        # file_record = metadata.copy()
            # self.files_meta[file_id] = file_record
            # self._save_metadata()

            # self._add_to_processing_queue(file_id)

        # file_path_obj = Path(item)
        # file_ext = file_path_obj.suffix.lower()

        # try:
        #     json_content = ""
        #     # 根据文件扩展名处理内容
        #     json_content = await process_file_to_json(item, params=params)
          
        #     chunks = []
        #     chunks = self.parse_json_into_embedding_chunks(json_content, file_id, filename, params)

        #     logger.info(f"Split {filename} into {len(chunks)} chunks")

        #     # 准备向量数据库插入的数据
        #     if chunks:
        #         documents = [chunk["content"] for chunk in chunks]
        #         embeddings = [chunk["embeddings"] for chunk in chunks]
        #         metadatas = [chunk["metadata"] for chunk in chunks]
        #         ids = [chunk["id"] for chunk in chunks]

        #         # 插入到 ChromaDB - 分批处理以避免超出 OpenAI 批次大小限制
        #         batch_size = 64  # OpenAI 的最大批次大小限制
        #         total_batches = (len(chunks) + batch_size - 1) // batch_size

        #         for i in range(0, len(chunks), batch_size):
        #             batch_documents = documents[i : i + batch_size]
        #             batch_embeddings = embeddings[i : i + batch_size]
        #             batch_metadatas = metadatas[i : i + batch_size]
        #             batch_ids = ids[i : i + batch_size]

        #             await asyncio.to_thread(
        #                 collection.add,
        #                 documents=batch_documents,
        #                 embeddings=batch_embeddings,
        #                 metadatas=batch_metadatas,
        #                 ids=batch_ids,
        #             )
        #             batch_num = i // batch_size + 1
        #             logger.info(f"Processed batch {batch_num}/{total_batches} for {filename}")

        #     logger.info(f"Inserted {content_type} {item} into Img_ChromaDB. Done.")

                # 更新状态为完成
                # self.files_meta[file_id]["status"] = "done"
                # self._save_metadata()
                # file_record["status"] = "done"

        # except Exception as e:
            # logger.error(f"处理{content_type} {item} 失败: {e}, {traceback.format_exc()}")
            # self.files_meta[file_id]["status"] = "failed"
            # self._save_metadata()
            # file_record["status"] = "failed"
            # raise e
                    
    
    async def aquery(self, db_id: str , query_text: Union[str, List[str]] = "", **kwargs) -> list[dict]:
        """异步查询知识库"""
        collection = await self._get_chroma_collection(db_id)
        if not collection:
            raise ValueError(f"Database {db_id} not found")

        try:
            top_k = kwargs.get("top_k", config.get("top_k", 10))
            similarity_threshold = kwargs.get("similarity_threshold", config.get("similarity_threshold", 0.0))
            where = kwargs.get("where")
            
            text_query_results = None
            if isinstance(query_text, str) and query_text == "":
                raise ValueError("Either query_text or query_embeddings must be provided")
            else:
                # 一行完成类型转换：如果是字符串转列表，否则保留列表（空值则转空列表）
                query_texts = [query_text] if isinstance(query_text, str) else (query_text or [])
                query_kwargs = {
                    "query_texts": query_texts,
                    "n_results": top_k,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    query_kwargs["where"] = where
                text_query_results = collection.query(           
                    **query_kwargs
                )

            documents = []
            metadatas = []
            distances = []
            # 处理文本查询结果
            # 遍历所有查询结果，支持多个查询文本
            if text_query_results and text_query_results.get("documents") and len(text_query_results["documents"]) > 0:
                for query_idx in range(len(text_query_results["documents"])):
                    if text_query_results["documents"][query_idx]:
                        documents.extend(text_query_results["documents"][query_idx])
                        if text_query_results.get("metadatas") and len(text_query_results["metadatas"]) > query_idx and text_query_results["metadatas"][query_idx]:
                            metadatas.extend(text_query_results["metadatas"][query_idx])
                        if text_query_results.get("distances") and len(text_query_results["distances"]) > query_idx and text_query_results["distances"][query_idx]:
                            distances.extend(text_query_results["distances"][query_idx])

            retrieved_chunks = []
            for i, doc in enumerate(documents):
                similarity = 1 - distances[i] if i < len(distances) else 1.0

                if similarity < similarity_threshold:
                    continue

                metadata = metadatas[i] if i < len(metadatas) else {}
                # 确保 file_id 在元数据中，并使用统一的键名
                if "full_doc_id" in metadata:
                    metadata["file_id"] = metadata.pop("full_doc_id")
                # chunk去重
                has_same_chunk_id = False
                for chunk in retrieved_chunks:
                    if chunk.get("metadata").get("chunk_id") == metadata.get("chunk_id"):
                        has_same_chunk_id = True
                        break
                if not has_same_chunk_id:
                    retrieved_chunks.append({"content": doc, "metadata": metadata, "score": similarity})

            logger.debug(f"ChromaDB query response: {len(retrieved_chunks)} chunks found (after similarity filtering)")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"ChromaDB query error: {e}, {traceback.format_exc()}")
            return []

    # async def aquery(self, db_id: str ,query_text: str = "" ,img_path: str = "",query_desc: str = "", **kwargs) -> list[dict]:
    #     """异步查询ChromaDB集合"""
    #     try:
    #         # 获取文本集合和图片集合
    #         text_collection = await self._get_chroma_collection(db_id)
    #         image_collection = await self._get_image_chroma_collection(db_id)
    #         if not text_collection and not image_collection:
    #             raise Exception(f"No collections found for db_id: {db_id}")

    #         # 处理查询参数
    #         top_k = kwargs.get("top_k", 10)
    #         similarity_threshold = kwargs.get("similarity_threshold", 0.0)

    #         results = []

    #         # 查询文本集合
    #         if text_collection and query_text:
    #             text_results = text_collection.query(
    #                 query_texts=[query_text],
    #                 n_results=top_k,
    #                 include=["documents", "metadatas", "distances"]
    #             )
                
    #             if text_results and text_results.get("documents"):
    #                 for i, doc in enumerate(text_results["documents"][0]):
    #                     if doc:
    #                         result = {
    #                             "content": doc,
    #                             "metadata": text_results["metadatas"][0][i] if text_results.get("metadatas") else {},
    #                             "score": 1-text_results["distances"][0][i] if text_results.get("distances") else 0.0
    #                         }
    #                         results.append(result)

    #         # 通过图片embedding查询
    #         if image_collection and img_path:
    #             # 获取图片嵌入
    #             image_embedding = get_image_embedding(img_path)
    #             if image_embedding is not None and len(image_embedding) > 0:
    #                 image_results = image_collection.query(
    #                     query_embeddings=[image_embedding],
    #                     n_results=top_k,
    #                     include=["documents", "metadatas", "distances"]
    #                 )
                    
    #                 if image_results and image_results.get("documents"):
    #                     for i, doc in enumerate(image_results["documents"][0]):
    #                         if doc:
    #                             result = {
    #                                 "content": doc,
    #                                 "metadata": image_results["metadatas"][0][i] if image_results.get("metadatas") else {},
    #                                 "score": 1-image_results["distances"][0][i] if image_results.get("distances") else 0.0
    #                             }
    #                             results.append(result)
    #         # 通过描述embedding查询
    #         if image_collection and query_desc:
    #             # 获取描述嵌入
    #             desc_embedding = get_text_embedding(query_desc)
    #             if desc_embedding is not None and len(desc_embedding) > 0:
    #                 desc_results = image_collection.query(
    #                     query_embeddings=[desc_embedding],
    #                     n_results=top_k,
    #                     include=["documents", "metadatas", "distances"]
    #                 )
                    
    #                 if desc_results and desc_results.get("documents"):
    #                     for i, doc in enumerate(desc_results["documents"][0]):
    #                         if doc:
    #                             result = {
    #                                 "content": doc,
    #                                 "metadata": desc_results["metadatas"][0][i] if desc_results.get("metadatas") else {},
    #                                 "score": 1-desc_results["distances"][0][i] if desc_results.get("distances") else 0.0    
    #                             }
    #                             results.append(result)
    #         # 去重和排序（先排序，再去重）
    #         seen_chunks = set()
    #         unique_results = []

    #         # 按距离排序（距离越小越相似）
    #         results.sort(key=lambda x: 1-x["score"])

    #         for result in results:
    #             chunk_id = result["metadata"].get("chunk_id")
    #             if chunk_id and chunk_id not in seen_chunks:
    #                 seen_chunks.add(chunk_id)
    #                 unique_results.append(result)

            

    #         # 应用相似度过滤
    #         filtered_results = []
            
    #         for result in unique_results:
    #             # 将距离转换为相似度（1 - 距离）
    #             similarity = result["score"]
    #             if similarity >= similarity_threshold:
    #                 result["similarity"] = similarity
    #                 filtered_results.append(result)

    #         return filtered_results[:top_k]
    #     except Exception as e:
    #         logger.error(f"Error querying ChromaDB for db_id {db_id}: {e}")
    #         return []


    async def delete_file(self, db_id: str, file_id: str) -> None:
        """删除文件"""
        collection = await self._get_chroma_collection(db_id)
        if collection:
            try:
                # 查找所有相关的chunks
                results = collection.get(where={"full_doc_id": file_id}, include=["metadatas"])

                # 删除所有相关chunks
                if results and results.get("ids"):
                    collection.delete(ids=results["ids"])
                    logger.info(f"Deleted {len(results['ids'])} chunks for file {file_id}")

            except Exception as e:
                logger.error(f"Error deleting file {file_id} from ChromaDB: {e}")

        # 删除文件记录
        if file_id in self.files_meta:
            del self.files_meta[file_id]
            self._save_metadata()

    async def get_file_basic_info(self, db_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        return {"meta": self.files_meta[file_id]}

    async def get_file_content(self, db_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks和lines）"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        # 使用 ChromaDB 获取chunks
        content_info = {"lines": []}
        collection = await self._get_chroma_collection(db_id)
        if collection:
            try:
                # 获取文档的所有chunks
                results = collection.get(where={"full_doc_id": file_id}, include=["documents", "metadatas"])

                # 构建chunks数据
                doc_chunks = []
                if results and results.get("ids"):
                    for i, chunk_id in enumerate(results["ids"]):
                        chunk_data = {
                            "id": chunk_id,
                            "content": results["documents"][i] if i < len(results["documents"]) else "",
                            "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {},
                            "chunk_order_index": results["metadatas"][i].get("chunk_index", i)
                            if i < len(results["metadatas"])
                            else i,
                        }
                        doc_chunks.append(chunk_data)

                # 按 chunk_order_index 排序
                doc_chunks.sort(key=lambda x: x.get("chunk_order_index", 0))
                content_info["lines"] = doc_chunks
                return content_info

            except Exception as e:
                logger.error(f"Failed to get file content from ChromaDB: {e}")
                content_info["lines"] = []
                return content_info

        return content_info

    async def get_file_info(self, db_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息+内容信息）- 保持向后兼容"""
        if file_id not in self.files_meta:
            raise Exception(f"File not found: {file_id}")

        # 合并基本信息和内容信息
        basic_info = await self.get_file_basic_info(db_id, file_id)
        content_info = await self.get_file_content(db_id, file_id)

        return {**basic_info, **content_info}
