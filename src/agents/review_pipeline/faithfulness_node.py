"""综述系统 - 忠实度校验节点（整篇校验版）

输入：review_markdown（整篇综述）+ extracted_data（证据库，按 paper_id 索引）
处理：从整篇拆出所有 [来源: paper_id] 论断 → 两层校验（规则层查 id 真伪 +
      LLM judge 查证据支撑）→ 按论断算造假率/不忠实率 → 三档判定
输出：verdict(pass/partial/mismatch) + 不达标论断详情，写入 faithfulness_report

与原 faithfulness_agent.py（按章节校验 + 定向重写）的区别：写作已改为一步成文，
综述是整篇无章节结构，故改为按整篇/按论断校验，partial 时整篇重写（非定向）。
两层校验机制、judge 并发独立实例、MAX_RETRY 防死循环均保留。原文件保留不动（v2 复用）。
"""
import re
import json
import asyncio
from typing import List, Tuple, Dict, Any

from autogen_agentchat.agents import AssistantAgent

from src.core.prompts import faithfulness_judge_prompt_review, faithfulness_revise_prompt_review
from src.core.model_client import create_model_client
from src.core.config import config
from src.core.state_models import ExecutionState, BackToFrontData
from src.utils.log_utils import setup_logger
from src.utils.llm_json import LLMJsonParseError, parse_llm_json_object

logger = setup_logger(__name__)

# 判定阈值（沿用原值，去掉章节维度的 bad_section_rate）
FAB_RATE_LIMIT = config.get_float("faithfulness_fab_rate_limit", 0.5)
UNFAITHFUL_RATE_LIMIT = config.get_float("faithfulness_unfaithful_rate_limit", 0.6)
MAX_RETRY = config.get_int("faithfulness_max_retry", 2)

_SRC_PATTERN = re.compile(r"\[来源[:：]\s*([^\]]+)\]")
FAITHFUL_THRESHOLD = 2  # judge 给分 <=1 视为不忠实
MIN_CLAIM_LEN = 10      # 论断最小长度：短于此视为残缺片段（如紧挨标点的"早期工作"），跳过不校验


def _parse_json_obj(s: str) -> dict:
    """从 judge 输出里稳健抠出 JSON。"""
    try:
        return parse_llm_json_object(s)
    except LLMJsonParseError:
        return {}


def _extract_claims(text: str) -> List[dict]:
    """从整篇综述里切出论断列表，每条记录定点替换所需的全部信息。

    返回 [{claim, ids, full_sentence, src_suffix}]：
    - claim：纯论断文本（已剔除 [来源] 标注），喂给 judge / 修订
    - ids：句内所有来源 id（去重保序）
    - full_sentence：含来源标注的完整原句，用于在原文里文本匹配替换（精准、不错位）
    - src_suffix：句内的来源标注原文（如 "[来源: x][来源: y]"），修订后拼回句尾，保住溯源链

    按句子切分（中文句末符 + 换行，不用英文句点——paper_id 含点会切碎标注）。
    一句多个 [来源] 视为一条 claim 引用多篇，不被切断。
    """
    claims = []
    for m in re.finditer(r"[^。！？\n]+", text):
        sentence = m.group(0)
        srcs = _SRC_PATTERN.findall(sentence)
        if not srcs:
            continue
        ids = []
        for raw in srcs:
            for x in re.split(r"[,，;；\s]+", raw):
                x = x.strip()
                if x and x not in ids:
                    ids.append(x)
        claim_text = _SRC_PATTERN.sub("", sentence).strip()
        if not claim_text or len(claim_text) < MIN_CLAIM_LEN:
            continue
        src_suffix = "".join(mm.group(0) for mm in _SRC_PATTERN.finditer(sentence))
        claims.append({
            "claim": claim_text,
            "ids": ids,
            "full_sentence": sentence,   # 完整原句（含标注），文本替换的定位锚
            "src_suffix": src_suffix,
        })
    return claims


def _format_evidence(ev: dict) -> str:
    """把一篇证据压成紧凑文本喂给 judge（对齐现有 4 抽取字段）。"""
    km = ev.get("key_methodology") or {}
    method = km.get("name") if isinstance(km, dict) else ""
    principle = km.get("principle") if isinstance(km, dict) else ""
    novelty = km.get("novelty") if isinstance(km, dict) else ""
    return (
        f"核心问题: {ev.get('core_problem','')}\n"
        f"方法: {method} | 原理: {principle} | 创新点: {novelty}\n"
        f"主要结果: {ev.get('main_results','')}\n"
        f"局限: {ev.get('limitations','')}"
    )


def _build_judge_agent():
    """每条 claim 用独立 judge 实例，避免并发评分时对话历史串扰。"""
    return AssistantAgent(
        name="faithfulness_judge",
        model_client=create_model_client("faithfulness-judge-model"),
        system_message=faithfulness_judge_prompt_review,
    )


async def _judge_claim(claim_text: str, evidence_text: str) -> dict:
    task = f"claim：{claim_text}\n\nevidence：\n{evidence_text}\n\n请按要求先计数再打分，只输出 JSON。"
    try:
        res = await _build_judge_agent().run(task=task)
        obj = _parse_json_obj(res.messages[-1].content)
        if "score" not in obj:
            obj["score"] = 1  # 解析失败保守判为部分不忠实
            obj["reason"] = "judge 输出解析失败，保守计为不达标"
        return obj
    except Exception as e:
        logger.error(f"judge 调用失败: {e}")
        return {"score": 1, "reason": f"judge 调用异常: {e}"}


def _build_revise_agent():
    """修订用独立 agent，prompt 与判分分离，避免自评偏袒。"""
    return AssistantAgent(
        name="faithfulness_reviser",
        model_client=create_model_client("faithfulness-judge-model"),
        system_message=faithfulness_revise_prompt_review,
    )


async def _revise_claim(claim_text: str, evidence_text: str, reason: str) -> str:
    """基于证据重写一条不忠实论断，返回改写后的句子（纯文本，不含来源标注）。"""
    task = (f"不忠实论断：{claim_text}\n\n引用证据：\n{evidence_text}\n\n"
            f"不忠实诊断：{reason}\n\n请基于证据改写这条论断，只输出改写后的一句话。")
    res = await _build_revise_agent().run(task=task)
    return _SRC_PATTERN.sub("", res.messages[-1].content).strip()


async def review_faithfulness_node(state) -> Dict[str, Any]:
    """整篇忠实度校验节点。"""
    state_queue = state["state_queue"]
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAITHFULNESS_CHECKING
        await state_queue.put(BackToFrontData(step=ExecutionState.FAITHFULNESS_CHECKING, state="initializing", data=None))

        review_md = current_state.review_markdown or ""

        # 证据库按 paper_id 建索引
        evidence_by_id: Dict[str, dict] = {}
        extracted = current_state.extracted_data
        if extracted is not None and hasattr(extracted, "papers"):
            for p in extracted.papers:
                d = p.model_dump() if hasattr(p, "model_dump") else p
                if d.get("paper_id"):
                    evidence_by_id[d["paper_id"]] = d
        valid_ids = set(evidence_by_id.keys())

        # 1. 整篇拆论断（每条带原文位置 start/end、来源标注 src_suffix）
        claims = _extract_claims(review_md)
        total_claims = len(claims)

        # 2. 规则层 + judge 层，记录每条的判定结果
        #    bad_claims: 待修订的不达标论断（含位置信息），good 直接通过
        bad_claims = []   # [{claim 各字段, score, reason}]
        fab_claims = 0
        unfaithful_claims = 0
        judge_targets = []  # 进 judge 的 (claim_dict)
        for c in claims:
            unknown = [i for i in c["ids"] if i not in valid_ids]
            if not c["ids"] or unknown:
                # 规则层：引用不存在的 id = 造假，无证据可修订，直接计入不达标
                fab_claims += 1
                c["score"] = 0
                c["reason"] = f"引用了不存在的来源 {unknown or '（无来源）'}（造假）"
                bad_claims.append(c)
            else:
                judge_targets.append(c)

        if judge_targets:
            results = await asyncio.gather(
                *[_judge_claim(c["claim"], _format_evidence(evidence_by_id[c["ids"][0]])) for c in judge_targets])
            for c, r in zip(judge_targets, results):
                if int(r.get("score", 1)) <= 1:
                    unfaithful_claims += 1
                    c["score"] = int(r.get("score", 1))
                    c["reason"] = r.get("reason", "")
                    bad_claims.append(c)

        # 3. mismatch 优先拦截：整体造假/不忠实率超阈值 = 检索召回与主题不匹配，
        #    根因在上游、修订个别句无意义，直接终止反馈（不进修订）。
        fab_rate = fab_claims / total_claims if total_claims else 0.0
        unfaithful_rate = unfaithful_claims / total_claims if total_claims else 0.0
        if total_claims > 0 and (fab_rate > FAB_RATE_LIMIT or unfaithful_rate > UNFAITHFUL_RATE_LIMIT):
            report = {
                "total_claims": total_claims, "fab_claims": fab_claims, "unfaithful_claims": unfaithful_claims,
                "fab_rate": round(fab_rate, 3), "unfaithful_rate": round(unfaithful_rate, 3),
                "detail": [f"- 论断「{c['claim'][:50]}」{c['score']}分：{c['reason']}" for c in bad_claims],
                "verdict": "mismatch",
            }
            current_state.faithfulness_report = report
            logger.info(f"忠实度校验：verdict=mismatch, {report}")
            await state_queue.put(BackToFrontData(
                step=ExecutionState.FAITHFULNESS_CHECKING, state="completed",
                data=f"忠实度校验：整体不匹配，造假率{fab_rate:.2f}，不忠实率{unfaithful_rate:.2f}"))
            return {"value": current_state}

        # 4. 定点修订：引用不存在来源的句子直接删除；有合法来源但不忠实的句子，
        #    用修订 prompt 改写 → 改写句回判分 prompt 验收 → 通过才收下。
        revisions = []   # (old_sentence, new_sentence) 验收通过的替换；new_sentence 为空表示删除
        still_bad = []    # 修订失败/无法修订，计入最终不达标
        await state_queue.put(BackToFrontData(step=ExecutionState.FAITHFULNESS_CHECKING, state="thinking",
                                              data=f"发现 {len(bad_claims)} 条不达标论断，开始定点修订\n"))
        for c in bad_claims:
            # 引用不存在 id 的句子没有可信证据可改写，直接删除比让模型硬圆更安全。
            if c["score"] == 0 and (not c["ids"] or any(i not in valid_ids for i in c["ids"])):
                revisions.append((c["full_sentence"], ""))
                continue
            ev_text = _format_evidence(evidence_by_id[c["ids"][0]])
            try:
                new_claim = await _revise_claim(c["claim"], ev_text, c["reason"])
                # 验收：改写句须非空、达最小长度，且回判分 prompt >=2 分才算修订成功。
                # 防止模型把无法支撑的论断改成空/极短残句却蒙混通过，替换后丢内容。
                if new_claim and len(new_claim) >= MIN_CLAIM_LEN:
                    verdict_obj = await _judge_claim(new_claim, ev_text)
                    if int(verdict_obj.get("score", 1)) >= FAITHFUL_THRESHOLD:
                        # 记 (原整句, 修订整句)：修订句拼回来源标注，构成可放回原文的完整句
                        revisions.append((c["full_sentence"], new_claim + c["src_suffix"]))
                    else:
                        still_bad.append(c)   # 改了仍不过 → 放弃，计入不达标
                else:
                    still_bad.append(c)   # 改写句空/过短 → 放弃
            except Exception as e:
                logger.error(f"修订失败: {e}")
                still_bad.append(c)

        # 5. 文本匹配替换：用原整句在全文里精准定位替换（不靠字符位置，避免区间错位）。
        #    只替换第一处出现（综述整句重复概率极低）。
        if revisions:
            for old_sentence, new_sentence in revisions:
                if old_sentence in review_md:
                    review_md = review_md.replace(old_sentence, new_sentence, 1)
                else:
                    logger.warning(f"修订句未在原文匹配到，跳过替换: {old_sentence[:40]}")
            current_state.review_markdown = review_md

        # 6. 最终判定（一轮，不循环；单条改不过即放弃）
        remaining_bad = len(still_bad)
        if remaining_bad == 0:
            verdict = "pass"   # 全部通过或全部修订成功
        else:
            verdict = "partial"   # 仍有改不好的 → 带警告放行

        report = {
            "total_claims": total_claims,
            "revised": len(revisions),
            "remaining_bad": remaining_bad,
            "fab_rate": round(fab_rate, 3),
            "unfaithful_rate": round(unfaithful_rate, 3),
            "detail": [f"- 论断「{c['claim'][:50]}」{c['score']}分：{c['reason']}" for c in still_bad],
            "verdict": verdict,
        }
        current_state.faithfulness_report = report

        logger.info(f"忠实度校验：verdict={verdict}, 修订{len(revisions)}条, 仍不达标{remaining_bad}条")
        # 推结构化结果给前端：汇总 + 仍不达标的论断详情（让用户感知哪些没改好、为什么）
        summary_payload = {
            "verdict": verdict,
            "total_claims": total_claims,
            "revised": len(revisions),
            "remaining_bad": remaining_bad,
            "detail": report["detail"],   # 仍不达标论断的「原句 + 分数 + 原因」列表
        }
        await state_queue.put(BackToFrontData(
            step=ExecutionState.FAITHFULNESS_CHECKING, state="completed",
            data=json.dumps(summary_payload, ensure_ascii=False)))
        return {"value": current_state}

    except Exception as e:
        import traceback
        err_msg = f"Faithfulness check failed: {str(e)}"
        logger.error(f"{err_msg}\n{traceback.format_exc()}")
        current_state.error.faithfulness_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.FAITHFULNESS_CHECKING, state="error", data=err_msg))
        return {"value": current_state}
