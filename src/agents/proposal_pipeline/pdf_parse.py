"""选题建议系统 - PDF 解析与切块工具。

职责：把已下载 PDF 解析为章节和 chunks，供论文画像与 RAG 入库共用。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz


SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "preliminaries": "background",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "proposed method": "method",
    "experiments": "experiment",
    "experiment": "experiment",
    "experimental results": "experiment",
    "evaluation": "experiment",
    "results": "experiment",
    "discussion": "discussion",
    "limitations": "limitation",
    "limitation": "limitation",
    "future work": "future_work",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "appendix": "appendix",
    "references": "references",
}

NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:[0-9]+(?:\.[0-9]+)*|[IVX]+)\.?\s+([A-Z][A-Za-z0-9 ,:/\-()]{2,90})\s*$"
)


@dataclass
class ParsedSection:
    section_key: str
    section_title: str
    page_start: int
    page_end: int
    text: str


@dataclass
class ParsedChunk:
    paper_id: str
    title: str
    section_key: str
    section_title: str
    page_start: int
    page_end: int
    chunk_index: int
    char_count: int
    text: str
    chunk_id: str = ""

    def metadata(self, paper: dict[str, Any], pdf_path: Path) -> dict[str, str | int | float | bool]:
        """生成可写入 Chroma 的 metadata。"""
        return sanitize_metadata(
            {
                "paper_id": self.paper_id,
                "title": self.title,
                "authors": paper.get("authors"),
                "year": paper.get("published"),
                "published_date": paper.get("published_date"),
                "primary_category": paper.get("primary_category"),
                "roles": paper.get("roles"),
                "matched_queries": paper.get("matched_queries"),
                "source": "pdf",
                "pdf_path": str(pdf_path),
                "section_key": self.section_key,
                "section_title": self.section_title,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "chunk_index": self.chunk_index,
                "char_count": self.char_count,
                "chunk_id": self.chunk_id,
                "full_doc_id": self.paper_id,
            }
        )

    def preview_dict(self, max_chars: int = 700) -> dict[str, Any]:
        """生成探针预览格式，避免把完整正文写入预览。"""
        data = asdict(self)
        data["text_preview"] = self.text[:max_chars]
        data.pop("text", None)
        if not data.get("chunk_id"):
            data.pop("chunk_id", None)
        return data


def normalize_line(line: str) -> str:
    """清洗 PDF 行文本。"""
    return re.sub(r"\s+", " ", line).strip()


def heading_to_key(line: str) -> tuple[str, str] | None:
    """识别章节标题，返回规范章节 key 和原始标题。"""
    clean = normalize_line(line)
    if not clean or len(clean) > 120:
        return None

    lower = clean.lower().strip(".:")
    if lower in SECTION_ALIASES:
        return SECTION_ALIASES[lower], clean

    match = NUMBERED_HEADING_RE.match(clean)
    if not match:
        return None

    title_part = match.group(1).lower().strip(".:")
    if title_part in SECTION_ALIASES:
        return SECTION_ALIASES[title_part], clean

    for alias, key in SECTION_ALIASES.items():
        if title_part.startswith(alias):
            return key, clean

    return None


def extract_sections(pdf_path: Path) -> list[ParsedSection]:
    """按启发式章节标题切分 PDF。"""
    doc = fitz.open(pdf_path)
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    reached_references = False

    for page_index in range(len(doc)):
        if reached_references:
            break
        page_number = page_index + 1
        text = doc.load_page(page_index).get_text("text")
        lines = [normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        for line in lines:
            heading = heading_to_key(line)
            if heading:
                if heading[0] == "references":
                    if current and current["lines"]:
                        current["page_end"] = page_number
                        sections.append(current)
                    reached_references = True
                    break
                if current and current["lines"]:
                    current["page_end"] = page_number
                    sections.append(current)
                section_key, section_title = heading
                current = {
                    "section_key": section_key,
                    "section_title": section_title,
                    "page_start": page_number,
                    "page_end": page_number,
                    "lines": [],
                }
                continue

            if current is None:
                current = {
                    "section_key": "front_matter",
                    "section_title": "Front Matter",
                    "page_start": page_number,
                    "page_end": page_number,
                    "lines": [],
                }
            current["page_end"] = page_number
            current["lines"].append(line)

    if current and current["lines"]:
        sections.append(current)
    doc.close()

    result: list[ParsedSection] = []
    for section in sections:
        text = "\n".join(section["lines"]).strip()
        if not text:
            continue
        result.append(
            ParsedSection(
                section_key=section["section_key"],
                section_title=section["section_title"],
                page_start=section["page_start"],
                page_end=section["page_end"],
                text=text,
            )
        )
    return result


def split_section(section: ParsedSection, max_chars: int = 3500, overlap: int = 300) -> list[str]:
    """长章节按字符长度二次切分。"""
    text = section.text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def build_chunks(
    paper: dict[str, Any],
    sections: list[ParsedSection],
    max_chars: int = 3500,
    overlap: int = 300,
) -> list[ParsedChunk]:
    """把章节转换为带元数据的 chunks。"""
    chunks: list[ParsedChunk] = []
    paper_id = str(paper.get("paper_id") or "")
    title = str(paper.get("title") or "")
    for section in sections:
        for index, chunk_text in enumerate(split_section(section, max_chars=max_chars, overlap=overlap)):
            chunks.append(
                ParsedChunk(
                    paper_id=paper_id,
                    title=title,
                    section_key=section.section_key,
                    section_title=section.section_title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=index,
                    char_count=len(chunk_text),
                    text=chunk_text,
                )
            )
    return chunks


def available_pdf_papers(
    manifest: dict[str, Any],
    paper_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """从 pdf_manifest 中筛选本地可用 PDF。"""
    papers = [
        paper
        for paper in manifest.get("papers", [])
        if paper.get("pdf_status") in ("cached", "downloaded")
        and paper.get("pdf_path")
        and Path(paper["pdf_path"]).exists()
    ]

    if paper_ids:
        wanted = set(paper_ids)
        papers = [paper for paper in papers if paper.get("paper_id") in wanted]
        missing = wanted - {str(paper.get("paper_id")) for paper in papers}
        if missing:
            raise ValueError(f"指定论文没有可用 PDF: {sorted(missing)}")

    if limit and limit > 0:
        papers = papers[:limit]

    if not papers:
        raise ValueError("没有可用 PDF 论文")
    return papers


def safe_id(value: str) -> str:
    """生成可用于向量库的稳定 ID。"""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:180] or "chunk"


def text_dedupe_hash(text: str) -> str:
    """生成正文去重哈希。"""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma metadata 只接受简单类型，复杂字段转成字符串。"""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, dict)):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean


def build_rag_document(paper: dict[str, Any], chunk: ParsedChunk) -> str:
    """把论文元信息写进 chunk 文本，避免召回片段脱离来源上下文。"""
    return "\n".join(
        [
            f"paper_id: {paper.get('paper_id')}",
            f"title: {paper.get('title')}",
            f"year: {paper.get('published')}",
            f"section_key: {chunk.section_key}",
            f"section_title: {chunk.section_title}",
            f"pages: {chunk.page_start}-{chunk.page_end}",
            f"chunk_id: {chunk.chunk_id}",
            "",
            chunk.text.strip(),
        ]
    ).strip()


def parse_pdfs_for_rag(
    papers: list[dict[str, Any]],
    max_chars: int = 3500,
    overlap: int = 300,
) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, Any]], dict[str, Any]]:
    """一次 PDF 解析同时产出 RAG documents、metadata、ids 和解析清单。"""
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    parse_reports: list[dict[str, Any]] = []
    seen_chunks: set[tuple[str, str, str]] = set()
    duplicate_chunk_count = 0

    for paper in papers:
        pdf_path = Path(paper["pdf_path"])
        sections = extract_sections(pdf_path)
        paper_chunk_count = 0
        paper_duplicate_count = 0
        section_reports = []

        for section_index, section in enumerate(sections):
            section_chunks = split_section(section, max_chars=max_chars, overlap=overlap)
            kept_section_chunk_count = 0
            section_reports.append(
                {
                    "section_key": section.section_key,
                    "section_title": section.section_title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "char_count": len(section.text),
                    "raw_chunk_count": len(section_chunks),
                    "chunk_count": 0,
                }
            )

            for chunk_index, chunk_text in enumerate(section_chunks):
                dedupe_key = (
                    str(paper.get("paper_id") or ""),
                    section.section_key,
                    text_dedupe_hash(chunk_text),
                )
                if dedupe_key in seen_chunks:
                    duplicate_chunk_count += 1
                    paper_duplicate_count += 1
                    continue
                seen_chunks.add(dedupe_key)

                chunk_id = safe_id(f"{paper.get('paper_id')}_{section_index}_{chunk_index}")
                chunk = ParsedChunk(
                    paper_id=str(paper.get("paper_id") or ""),
                    title=str(paper.get("title") or ""),
                    section_key=section.section_key,
                    section_title=section.section_title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    text=chunk_text,
                    chunk_id=chunk_id,
                )
                documents.append(build_rag_document(paper, chunk))
                metadatas.append(chunk.metadata(paper, pdf_path))
                ids.append(chunk_id)
                paper_chunk_count += 1
                kept_section_chunk_count += 1

            section_reports[-1]["chunk_count"] = kept_section_chunk_count

        parse_reports.append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "pdf_path": str(pdf_path),
                "section_count": len(sections),
                "chunk_count": paper_chunk_count,
                "duplicate_chunk_count": paper_duplicate_count,
                "sections": section_reports,
            }
        )

    dedupe_summary = {
        "duplicate_chunk_count": duplicate_chunk_count,
        "dedupe_rule": "paper_id + section_key + normalized_text_sha256",
    }
    return documents, metadatas, ids, parse_reports, dedupe_summary
