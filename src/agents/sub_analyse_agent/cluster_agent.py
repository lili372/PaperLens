#!/usr/bin/env python3
"""
论文分析智能体 - 聚类功能实现
包含嵌入向量生成、KMeans聚类和LLM主题描述生成
"""

import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from src.core.model_client import create_default_client, create_subanalyse_cluster_model_client, create_cluster_embedding_client
from src.core.prompts import clustering_agent_prompt
from src.agents.reading_agent import ExtractedPaperData, ExtractedPapersData
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dataclasses import dataclass
from src.utils.log_utils import setup_logger

# 配置日志
logger = setup_logger(__name__)


@dataclass
class PaperCluster:
    """论文聚类结果"""
    cluster_id: int
    papers: List[Dict[str, Any]]
    theme_description: str
    keywords: List[str]
    centroid_vector: np.ndarray = None
    member_embeddings: np.ndarray = None  # 成员论文的向量，用于起名时挑最接近质心的代表作

class PaperClusterAgent:
    """论文聚类智能体"""
    
    def __init__(self, model_client=None):
        """初始化聚类智能体"""
        self.model_client = create_subanalyse_cluster_model_client()
        self.clustering_agent = AssistantAgent(
            name="clustering_agent",
            model_client= self.model_client,
            system_message = clustering_agent_prompt
        )


    def get_embedding(self, text: Union[str, List[str]]) -> list[float]:
        client = create_cluster_embedding_client()
        if isinstance(text, str):
            text = [text]
        # Qwen(DashScope) embedding 单次 batch 上限 10 条，超过会报 400，这里分批
        res = []
        BATCH = 10
        for i in range(0, len(text), BATCH):
            response = client.embeddings.create(
                model=client.default_headers["X-Model"],
                input=text[i:i + BATCH],
                dimensions=1024
            )
            for tmp in response.data:
                res.append(tmp.embedding)
        return res

    def prepare_text_for_embedding(self, paper: Dict[str, Any]) -> str:
        """准备用于生成嵌入向量的文本"""
        text_parts = []
        
        # 核心问题
        if paper.get('core_problem'):
            text_parts.append(f"Problem: {paper['core_problem']}")
            
        # 方法论
        if paper.get('key_methodology'):
            methodology = paper['key_methodology']
            text_parts.append(f"Method: {methodology.get('name', '')} - {methodology.get('principle', '')}")
            
        # 主要结果
        if paper.get('main_results'):
            if isinstance(paper['main_results'], list):
                results = "; ".join(paper['main_results'])
            else:
                results = str(paper['main_results'])
            text_parts.append(f"Results: {results}")

        return " ".join(text_parts)
    
    def generate_embeddings(self, papers: List[Dict[str, Any]]) -> np.ndarray:
       
        texts = [self.prepare_text_for_embedding(paper) for paper in papers]
        
        embeddings = self.get_embedding(texts)
        
        return np.array(embeddings)
        
    def determine_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 5) -> int:
        """使用肘部法则确定最佳聚类数量"""
        if len(embeddings) <= 2:
            return 1
            
        max_clusters = min(max_k, len(embeddings) - 1)
        if max_clusters == 1:
            return 1
            
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            if k <= len(embeddings):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
        
        # 肘部法则：找"降幅由陡变平的拐点"。
        # 注意不能用"一阶差值最大处"——inertia 从 k=1 到 k=2 必然降得最多，
        # 那样几乎永远输出 k=2，不是真正的肘部。正确做法是看二阶差分
        # （降幅的变化量）最大处，即下降速度骤减的那个 k。
        if len(inertias) >= 3:
            # 一阶差分：每增加一个簇带来的 inertia 降幅
            diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            # 二阶差分：降幅的减少量，越大说明"再多分一类已不划算"
            second_diffs = [diffs[i-1] - diffs[i] for i in range(1, len(diffs))]
            # second_diffs[j] 对应 k=j+2 这个拐点
            optimal_k = second_diffs.index(max(second_diffs)) + 2
            return min(optimal_k, max_clusters)
        else:
            return min(2, max_clusters)
    
    def cluster_papers(self, papers: List[Dict[str, Any]]) -> List[PaperCluster]:
        """对论文进行聚类"""
        if not papers:
            return []
            
        # 生成嵌入向量
        embeddings = self.generate_embeddings(papers)
        
        # 确定聚类数量
        n_clusters = self.determine_optimal_clusters(embeddings)
        
        if n_clusters == 1 or len(papers) <= n_clusters:
            # 所有论文在一个聚类中
            return [PaperCluster(
                cluster_id=0,
                papers=papers,
                theme_description="General Research Papers",
                keywords=["general"],
                centroid_vector=np.mean(embeddings, axis=0),
                member_embeddings=embeddings
            )]
        
        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 构建聚类结果
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_papers = [
                papers[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            clusters.append(PaperCluster(
                cluster_id=cluster_id,
                papers=cluster_papers,
                theme_description="",
                keywords=[],
                centroid_vector=centroid,
                member_embeddings=cluster_embeddings
            ))
        
        return clusters
    
    def parse_llm_response(self, response: str) -> Tuple[str, List[str]]:
        """解析LLM响应，提取主题描述和关键词"""
        try:
            import re
            
            # 使用正则表达式匹配主题描述，支持中英文冒号、空格等变化
            theme_pattern = r'主题描述\s*[:：]\s*\[([^\]]+)\]'
            theme_match = re.search(theme_pattern, response, re.IGNORECASE)
            
            # 使用正则表达式匹配关键词，支持中英文冒号、空格等变化
            keywords_pattern = r'关键词\s*[:：]\s*\[([^\]]+)\]'
            keywords_match = re.search(keywords_pattern, response, re.IGNORECASE)
            
            # 提取主题描述
            if theme_match:
                theme_description = theme_match.group(1).strip()
                # 清理可能的额外引号或空格
                theme_description = re.sub(r'^["\']|["\']$', '', theme_description).strip()
            else:
                # 尝试其他可能的格式变体
                alt_theme_patterns = [
                    r'主题\s*[:：]\s*([^\n]+)',
                    r'theme\s*[:：]\s*([^\n]+)',
                    r'主题描述\s*[:：]\s*([^\n]+)'
                ]
                
                theme_description = "未分类研究主题"
                for pattern in alt_theme_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        theme_description = match.group(1).strip()
                        break
            
            # 提取关键词
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                # 支持逗号、分号、空格等多种分隔符
                keywords = []
                for separator in [',', ';', '，', '；']:
                    if separator in keywords_str:
                        keywords = [kw.strip() for kw in keywords_str.split(separator) if kw.strip()]
                        break
                
                # 如果没有找到分隔符，尝试按空格分割
                if not keywords:
                    keywords = [kw.strip() for kw in keywords_str.split() if kw.strip()]
                
                # 清理每个关键词中的额外引号
                keywords = [re.sub(r'^["\']|["\']$', '', kw).strip() for kw in keywords]
            else:
                # 尝试其他可能的关键词格式
                alt_keywords_patterns = [
                    r'关键词\s*[:：]\s*([^\n]+)',
                    r'keywords\s*[:：]\s*([^\n]+)',
                    r'key\s+words\s*[:：]\s*([^\n]+)'
                ]
                
                keywords = ["research"]
                for pattern in alt_keywords_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        keywords_str = match.group(1).strip()
                        keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                        break
            
            # 确保至少有一个关键词
            if not keywords:
                keywords = ["research"]
            
            # 限制关键词数量
            keywords = keywords[:5]
            
            return theme_description, keywords
            
        except Exception as e:
            logger.error(f"解析LLM响应时出错:\n {e}")
            return "未分类研究主题", ["research"]
    
    async def generate_cluster_theme(self, cluster: PaperCluster) -> str:
        """使用LLM为聚类生成主题描述和关键词"""
        try:
            # 起名只看最能代表本簇的论文：挑离质心最近的若干篇。
            # 不取"前N篇"（任意顺序、可能抽到边缘论文把主题带偏），也不全取
            # （边缘论文会稀释主题、且大簇 token 过多）——起名要的是抓核心特征，
            # 典型样本足矣；覆盖全部是深度分析那一步的职责。
            TOP_N = 5
            if cluster.member_embeddings is not None and cluster.centroid_vector is not None:
                dists = np.linalg.norm(cluster.member_embeddings - cluster.centroid_vector, axis=1)
                rep_idx = np.argsort(dists)[:TOP_N]
                rep_papers = [cluster.papers[i] for i in rep_idx]
            else:
                rep_papers = cluster.papers[:TOP_N]

            paper_summaries = []
            for paper in rep_papers:
                summary = {
                    "problem": paper.get("core_problem", ""),
                    "method": paper.get("key_methodology", {}).get("name", ""),
                    "results": paper.get("main_results", "")
                }
                paper_summaries.append(summary)
            
            prompt = f"""
                基于以下论文信息，为这一类论文提炼它们共同的研究方向主题(必须生成一个主题，不能为空)和3-5个关键词：

                论文信息：
                {json.dumps(paper_summaries, ensure_ascii=False, indent=2)}

                请提供：
                1. 一个能作为综述章节标题的研究方向主题（简洁的名词短语，不超过15字，不要写成句子）。
                   要用最能体现这批论文技术特色的具体术语（如具体的方法范式、问题设定、技术手段），
                   不要用"方法研究""技术应用""算法与优化""方法与机制"这类放在任何方向都成立的宽泛词。
                2. 3-5个关键词（用逗号分隔）

                格式：
                主题描述：[主题描述]
                关键词：[关键词1, 关键词2, 关键词3]
            """
            response = await self.clustering_agent.run(task=prompt)
            
            # 解析LLM响应
            theme_description, keywords = self.parse_llm_response(response.messages[-1].content)
            return theme_description, keywords
                
        except Exception as e:
            logger.error(f"生成聚类主题时出错: \n{e}")
            # 必须返回与正常路径一致的 (theme, keywords) 元组——
            # 调用方用 `theme, kws = await generate_cluster_theme(...)` 解包，
            # 若只返回单个字符串会触发 unpack 错误，让异常处理本身再次崩溃。
            # 起名失败不应中断整个聚类流程，用默认名继续。
            return "未分类研究主题", ["research"]
    

    async def run_clustering_analyse(self, papers_data: Dict[str, Any]) -> List[PaperCluster]:
        """运行完整的聚类分析"""
        papers = papers_data.get("papers", [])
        
        if not papers:
            return {"error": "没有论文数据可供分析"}
        
        logger.info(f"开始对 {len(papers)} 篇论文进行聚类分析...")
        
        # 执行聚类
        clusters = self.cluster_papers(papers)
        
        # 为每个聚类生成主题和关键词
        results = []
        for cluster in clusters:
            theme_description, keywords = await self.generate_cluster_theme(cluster)
            paperCluster = PaperCluster(
                cluster_id=cluster.cluster_id,
                papers=cluster.papers,
                theme_description=theme_description,
                keywords=keywords)  
            results.append(paperCluster)
        
        return results
    def run(self, papers_data: ExtractedPapersData):
        """统一接口方法"""
        papers = papers_data.model_dump()
        return self.run_clustering_analyse(papers)

async def main():
    """主测试函数"""
    # 测试数据
    test_papers = papers
    
    # 创建聚类智能体
    agent = PaperClusteringAgent()
    
    # 运行聚类分析
    results = await agent.run_cluster_analyse(test_papers)
    
    # 输出结果
    print("=" * 60)
    print("论文聚类分析结果")
    print("=" * 60)
    print(f"总论文数: {results['total_papers']}")
    print(f"聚类数量: {results['total_clusters']}")
    print()
    
    for cluster in results['clusters']:
        print(f"聚类 {cluster['cluster_id'] + 1}:")
        print(f"  主题: {cluster['theme']}")
        print(f"  关键词: {', '.join(cluster['keywords'])}")
        print(f"  论文数: {cluster['paper_count']}")
        print("  包含论文:")
        for paper in cluster['papers']:
            print(f"    - {paper['paper_id']}: {paper['title']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())