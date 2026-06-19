"""大模型 JSON 输出解析兜底工具。

这里只处理格式层面的常见偏差：代码块包裹、单引号字面量、正文前后混入说明、
对象/数组被包在长文本中。具体字段是否完整、枚举是否合法，仍由业务节点校验。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


class LLMJsonParseError(ValueError):
    """大模型输出无法解析为目标 JSON 类型。"""


def strip_json_fence(value: Any) -> str:
    """去掉模型常见的 ```json 代码块包裹。"""
    text = str(value).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _try_parse(text: str) -> Any:
    """依次尝试标准 JSON 和 Python 字面量解析。"""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception as exc:
        raise LLMJsonParseError(f"无法解析 JSON: {exc}") from exc


def _extract_enclosed(text: str, left: str, right: str) -> str | None:
    start = text.find(left)
    end = text.rfind(right)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_llm_json(value: Any) -> Any:
    """把模型输出尽量解析为 JSON 兼容对象。"""
    if isinstance(value, (dict, list)):
        return value

    text = strip_json_fence(value)
    try:
        return _try_parse(text)
    except LLMJsonParseError:
        pass

    candidates = [
        _extract_enclosed(text, "{", "}"),
        _extract_enclosed(text, "[", "]"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return _try_parse(candidate)
        except LLMJsonParseError:
            continue

    raise LLMJsonParseError("模型输出中未找到可解析的 JSON 对象或数组")


def parse_llm_json_object(value: Any) -> dict[str, Any]:
    """解析模型输出为 JSON 对象。"""
    parsed = parse_llm_json(value)
    if not isinstance(parsed, dict):
        raise LLMJsonParseError("模型输出不是 JSON 对象")
    return parsed


def parse_llm_json_array(value: Any) -> list[Any]:
    """解析模型输出为 JSON 数组。"""
    parsed = parse_llm_json(value)
    if not isinstance(parsed, list):
        raise LLMJsonParseError("模型输出不是 JSON 数组")
    return parsed
