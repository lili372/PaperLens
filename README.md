<h1 align="center">PaperLens</h1>
<p align="center">面向科研阅读与选题的论文分析工具</p>

---

## 这是什么

PaperLens 面向研究生和科研初学者，帮助你在一个研究方向上更快完成两件事：

1. **综述分析**：输入研究方向，自动检索 arXiv 论文，生成一篇带来源标注的导航型综述。
2. **研究方向推荐**：输入一个领域，系统会检索相关论文、分析可组合的方法模块，并输出候选选题方向、证据依据、风险提示和建议精读论文。

它不是替代精读论文的工具，而是帮你先完成“该看哪些论文、这个方向有哪些切入点、哪些组合不太可靠”的前置判断。

## 核心功能

- **综述分析**：从研究主题出发，完成论文检索、摘要抽取、方向聚类、综述生成和忠实度校验。
- **研究方向推荐**：从领域输入出发，生成检索计划，下载并筛选论文 PDF，抽取论文画像，生成候选选题并做证据查证与可做性审查。
- **人工确认节点**：在检索计划、召回论文、进入分析的论文范围等关键步骤暂停，允许用户确认或调整。
- **证据可追溯**：报告中的关键结论会绑定论文 ID 或 RAG 证据，便于回到原论文核查。
- **Markdown 导出**：综述报告和研究方向报告都支持下载为 Markdown。

## 两种使用模式

### 综述分析

适合在进入一个陌生领域时快速建立全局认识。

流程：

```text
输入研究方向
  -> 生成检索关键词
  -> arXiv 检索
  -> 抽取论文要点
  -> 聚类研究方向
  -> 生成导航型综述
  -> 忠实度校验与修订
  -> 展示并下载 Markdown
```

### 研究方向推荐

适合在已有大致方向后，判断哪些论文值得优先精读、哪些方法模块可能形成新的选题组合。

流程：

```text
输入研究方向
  -> 生成或填写检索计划
  -> 搜索论文并下载 PDF
  -> 确认实际进入分析的论文
  -> 生成论文画像
  -> 生成候选研究方向
  -> RAG 证据查证
  -> 可做性审查与必要修改
  -> 输出研究方向报告
```

研究方向推荐支持两种入口：

- **普通模式**：输入自然语言需求，由系统生成 `base_query`、`module_terms` 和时间范围。
- **专家模式**：直接填写 `base_query`、`module_terms` 和时间范围。

## 输出内容

研究方向报告通常包含：

- 推荐研究方向标题
- 为什么值得做
- 来源论文与证据强度
- 主要风险和收窄建议
- 建议优先精读的论文
- 其他可考虑或暂缓的方向

## 技术栈

- **后端**：Python、FastAPI、AutoGen、ChromaDB、arXiv API
- **前端**：Vue 3、Vite、SSE
- **模型接口**：DeepSeek 兼容接口用于生成与审查，DashScope / 通义千问用于 embedding

## 快速开始

### 1. 准备环境变量

复制 `.env.example` 为 `.env`，填入自己的 API Key：

```bash
cp .env.example .env
```

需要配置：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DASHSCOPE_API_KEY=你的 DashScope API Key
ARXIV_API_URL=http://export.arxiv.org/api/query
```

### 2. 启动后端

需要 Python 3.12。

```bash
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements_frozen.txt
python main.py
```

后端默认运行在：

```text
http://localhost:8000
```

### 3. 启动前端

需要 Node.js 18+。

```bash
cd web
npm install
npm run dev
```

浏览器打开终端提示的地址，默认是：

```text
http://localhost:5173
```

## 本地数据

运行过程中会产生本地缓存和报告文件：

```text
data/
```

其中 V2 研究方向推荐的运行缓存默认保存在：

```text
data/proposal_runs/
```

这些运行数据不会提交到 Git。

## 注意事项

- 论文检索和 PDF 下载依赖 arXiv 网络状态。
- PDF 下载失败、论文过长或综述类论文会被跳过，并在前端显示统计信息。
- 生成报告用于辅助选题判断，不能替代人工精读和实验验证。

