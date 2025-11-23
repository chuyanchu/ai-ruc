from __future__ import annotations

import html
import hashlib
from datetime import datetime
from textwrap import dedent
import os
import json
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.parser import parse as dtparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILES = ("rag_outputs.jsonl", "rag_outputs.json")
SNIPPET_TAG_RE = re.compile(r"【[^】]+】")


def clean_snippet_text(text: str) -> str:
    text = (text or "").strip()
    text = SNIPPET_TAG_RE.sub("", text)
    text = text.replace("。。", "。")
    return " ".join(text.split())


def _iter_json_lines(file_obj):
    """Yield JSON objects from newline-delimited files, skipping broken rows."""
    for line in file_obj:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


@st.cache_data(show_spinner="正在加载知识库数据...")
def load_data():
    """
    从本地文件（或GitHub克隆后的文件系统）加载数据
    并利用 Streamlit 缓存避免每次刷新都重新读取
    """
    docs = []
    notices = []
    
    # 获取当前脚本所在的绝对路径，确保路径准确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = None
    for filename in DATA_FILES:
        candidate = os.path.join(current_dir, filename)
        if os.path.exists(candidate):
            file_path = candidate
            break

    if not file_path:
        expected = " 或 ".join(DATA_FILES)
        st.error(f"错误：找不到文件 {expected}。请确保该文件已上传到 GitHub 仓库的同一目录下。")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith(".jsonl"):
                records = _iter_json_lines(f)
            else:
                try:
                    payload = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    records = _iter_json_lines(f)
                else:
                    if isinstance(payload, list):
                        records = payload
                    elif isinstance(payload, dict):
                        records = [payload]
                    else:
                        records = []

            for item in records:
                meta = item.get("meta", {})
                rag = item.get("rag", {})
                rag_question = (rag.get("query") or "").strip()

                # 1. 处理文档数据 (DOCS)
                sections = [
                    f"【标题】{meta.get('title', '')}",
                ]
                if rag_question:
                    sections.append(f"【提问】{rag_question}")
                sections.extend(
                    [
                        f"【摘要】{rag.get('answer', '')}",
                        f"【链接】{meta.get('detail_url', '')}",
                    ]
                )
                doc_content = "\n".join(sections)
                raw_answer = (rag.get("answer") or "").strip()
                pure_text = clean_snippet_text(raw_answer) if raw_answer else ""
                if not pure_text:
                    pure_text = clean_snippet_text(doc_content)
                docs.append({
                    "id": str(meta.get("id")),
                    "title": meta.get("title", "无标题"),
                    "dept": "信息学院", 
                    "url": meta.get("detail_url", "#"),
                    "content": doc_content,
                    "pure_text": pure_text,
                    "raw_answer": rag.get("answer", ""),
                    "source_query": rag_question,
                })
                
                # 2. 处理通知数据 (NOTICES)
                notices.append({
                    "id": str(meta.get("id")),
                    "title": meta.get("title", "无标题"),
                    "dept": "信息学院",
                    "publish_time": meta.get("date", ""),
                    "deadline": None,
                    "tags": ["学术", "动态"],
                    "content": rag.get("answer", ""),
                    "url": meta.get("detail_url", "#")
                })
    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return [], []
    
    return docs, notices

DOCS, NOTICES = load_data()

# 补充静态数据（保留原有的静态数据逻辑，防止数据太少）
STATIC_DOCS = [
    {
        "id": "finance_2025_reimburse",
        "title": "2025年度经费报销规范(财务处).pdf",
        "dept": "财务处",
        "url": "#",
        "content": "报销范围: 会议差旅、科研购置... 额度上限: A 类项目单笔不超过 12,000 元...",
        "pure_text": "标准流程涵盖会议差旅、科研购置等，A 类项目单笔报销上限 12,000 元。",
        "raw_answer": "财务处报销规范涉及会议差旅等，A类限额1.2万。",
        "source_query": "",
    }
    # ... 你可以保留之前其他的静态数据 ...
]
if DOCS:
    DOCS.extend(STATIC_DOCS)


# 应用的标题与宣传文案
APP_TITLE = "RUC Connect · Executive Intelligence Cockpit"
APP_SUBTITLE = "AI驱动的一体化校园服务平台, 助力全球高校实现治理跃迁。"
APP_TAGLINE = "One campus. Every stakeholder. Decisions in minutes."


# 示例文档数据（仅在真实数据缺失时兜底展示）
FALLBACK_DOCS = [
    {
        "id": "finance_2025_reimburse",
        "title": "2025年度经费报销规范(财务处).pdf",
        "dept": "财务处",
        "url": "https://intra.ruc.edu.cn/finance/2025-reimburse.pdf",
        "content": (
            "报销范围: 会议差旅、科研购置、学术交流等; 超出范围需提交专项说明。"
            "额度上限: A 类项目单笔不超过 12,000 元, B 类项目单笔不超过 8,000 元。"
            "流程节点: 线上提单 → 院系审核 → 财务复核 → 银行入账。"
            "所需材料: 发票原件、经费批准文件、电子报销单、支付凭证。"
        ),
        "pure_text": (
            "可以报销会议差旅、科研购置等，A 类单笔≤12,000 元、B 类≤8,000 元；流程为线上提单、院系审核、财务复核、入账。"
        ),
    },
    {
        "id": "research_2025_core_journals",
        "title": "2025 核心期刊认定清单(科研处).xlsx",
        "dept": "科研处",
        "url": "https://intra.ruc.edu.cn/research/2025-corejournals.xlsx",
        "content": (
            "A 类: Journal of Public Administration Research and Theory, Government Information Quarterly 等。"
            "信息资源管理学报、Government Information Quarterly、智库管理评论等列为重点推荐。"
            "B 类: 管理世界、中国行政管理、电子政务研究等。"
        ),
        "pure_text": (
            "A 类重点为 Journal of Public Administration Research and Theory、Government Information Quarterly 等，B 类包括管理世界、中国行政管理等。"
        ),
    },
    {
        "id": "student_handbook_medical",
        "title": "学生医保与就医指引(学生处).pdf",
        "dept": "学生处",
        "url": "https://intra.ruc.edu.cn/student/2025-medical.pdf",
        "content": (
            "医保报销比例: 校医院 70%, 合作医院 50%-65%, 个人自费部分按政策结算。"
            "就医流程: 线上预约 → 自助机取号 → 医生诊疗 → 药房取药。"
            "报销材料: 挂号票据、诊断证明、药品清单、医保卡复印件。"
        ),
        "pure_text": (
            "校医院医保报销 70%，合作医院 50%-65%；就医流程为线上预约、取号、诊疗、取药，并准备发票、诊断证明等材料。"
        ),
    },
    {
        "id": "it_print_service",
        "title": "校园打印与自助终端服务指引.docx",
        "dept": "信息中心",
        "url": "https://intra.ruc.edu.cn/it/print-service.docx",
        "content": (
            "打印网点: 艺术楼一层、图书馆负一层、信息楼一层均支持自助打印。"
            "支持纸型: A4 / A3, 黑白 0.2 元/页, 彩色 0.8 元/页, 可绑定校园卡支付。"
            "开放时间: 08:00-22:00, 周末正常开放; 提供远程上传与批量打印。"
        ),
        "pure_text": (
            "艺术楼、图书馆、信息楼提供自助打印，黑白 0.2 元、彩色 0.8 元，可绑定校园卡并支持远程上传、批量打印。"
        ),
    },
]

# 示例通知公告（仅在真实数据缺失时兜底展示）
FALLBACK_NOTICES = [
    {
        "id": "nsfc_2025_fund",
        "title": "国家自科基金数字治理项目 2025 年度申报通知",
        "dept": "科研处",
        "publish_time": "2025-10-20",
        "deadline": "2025-11-03",
        "tags": ["基金申报", "数字治理", "青年教师"],
        "content": "聚焦数字治理、数据要素流通等方向, 设置重点/面上/青年项目, 请于 11 月 3 日前完成线上申报。",
        "url": "https://intra.ruc.edu.cn/research/notice-nsfc-2025.html",
    },
    {
        "id": "law_ai_lecture",
        "title": "法学院讲座: AIGC 时代的数据合规与协同",
        "dept": "法学院",
        "publish_time": "2025-10-25",
        "deadline": "2025-11-01",
        "tags": ["讲座", "数字治理", "合规", "AIGC"],
        "content": "主讲人: 刘昊宇教授, 议题覆盖 AI 合规架构、数据伦理审查与跨境合作案例, 欢迎师生报名参与。",
        "url": "https://law.ruc.edu.cn/events/aigc-compliance.html",
    },
    {
        "id": "bus_timetable",
        "title": "校园巴士冬季时刻调整(后勤处)",
        "dept": "后勤处",
        "publish_time": "2024-12-01",
        "deadline": None,
        "tags": ["校园巴士", "出行"],
        "content": "工作日 07:00-21:30 每 15 分钟一班, 周末 08:00-20:00 每 20 分钟一班, 请提前 5 分钟抵达站点候车。",
        "url": "https://intra.ruc.edu.cn/logistics/bus-winter.html",
    },
]

if not DOCS:
    DOCS = list(FALLBACK_DOCS)

if not NOTICES:
    NOTICES = list(FALLBACK_NOTICES)

# 示例查询日志, 用于洞察统计
QUERY_LOG = pd.DataFrame(
    [
        {"q": "校医院挂号", "hits": 42, "ok": True},
        {"q": "学生医保报销流程", "hits": 26, "ok": True},
        {"q": "宿舍空调报修渠道", "hits": 18, "ok": False},
        {"q": "校园卡补办进度", "hits": 15, "ok": False},
        {"q": "基金申报截止时间", "hits": 12, "ok": True},
        {"q": "科研项目经费报销", "hits": 11, "ok": True},
        {"q": "博士生助研津贴发放", "hits": 9, "ok": True},
        {"q": "打印彩色收费标准", "hits": 8, "ok": True},
        {"q": "图书馆期末延时开放", "hits": 7, "ok": True},
        {"q": "心理咨询预约渠道", "hits": 7, "ok": True},
        {"q": "运动场馆预约规则", "hits": 6, "ok": False},
        {"q": "校园巴士寒假班次", "hits": 6, "ok": False},
        {"q": "实验室开放时间", "hits": 5, "ok": True},
        {"q": "交换生报名条件", "hits": 5, "ok": False},
        {"q": "宿舍电费自助缴费", "hits": 4, "ok": True},
        {"q": "校园网账号解锁", "hits": 4, "ok": False},
        {"q": "毕业证书领取流程", "hits": 3, "ok": True},
        {"q": "研究生培养方案", "hits": 3, "ok": False},
        {"q": "实习证明盖章地点", "hits": 2, "ok": False},
        {"q": "人工智能课程调课", "hits": 1, "ok": False},
    ]
)
# -------------------- 核心工具函数 --------------------


# 将原始内容裁剪到指定长度, 保持展示整洁
def smart_truncate(text: str, max_len: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _vectorizer_signature(docs: list[dict[str, str]]):
    return tuple((doc["id"], doc["content"]) for doc in docs)


@st.cache_resource(show_spinner="正在构建向量索引...")
def _build_tfidf_from_signature(signature: tuple[tuple[str, str], ...]):
    if not signature:
        return None, None
    texts = [content for _, content in signature]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_df=0.95,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# 构建 TF-IDF 向量用于检索（缓存结果以避免重复计算）
def build_tfidf_corpus(docs: list[dict[str, str]]):
    signature = _vectorizer_signature(docs)
    if not signature:
        return None, None
    return _build_tfidf_from_signature(signature)


# 基于向量相似度检索最相关的文档, 支持阈值与低置信回退
def rag_search(
    query: str,
    docs: list[dict[str, str]],
    vectorizer,
    matrix,
    topk: int = 3,
    min_score: float = 0.12,
) -> tuple[list[dict[str, str]], bool]:
    if not docs or vectorizer is None or matrix is None:
        return [], False

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    indices = np.argsort(-sims)
    results = []
    fallback = []
    for idx in indices:
        score = float(sims[idx])
        doc = docs[idx]
        snippet_source = doc.get("pure_text") or doc.get("raw_answer") or doc.get("content", "")
        snippet_text = smart_truncate(clean_snippet_text(snippet_source), 180)
        entry = {
            "id": doc["id"],
            "title": doc["title"],
            "dept": doc["dept"],
            "url": doc["url"],
            "score": score,
            "chunk": smart_truncate(doc["content"], 220),
            "content": doc["content"],
            "source_query": doc.get("source_query", ""),
            "snippet": snippet_text,
            "pure_text": doc.get("pure_text") or snippet_source,
        }
        if len(fallback) < topk:
            fallback.append(entry)
        if score >= min_score:
            results.append(entry)
            if len(results) >= topk:
                break

    if results:
        return results, False
    return fallback, True


# 汇总检索结果, 生成易读回答
def synthesize_answer(query: str, hits: list[dict[str, str]]) -> str:
    """
    生成上方的主回答卡片，模拟 AI 总结语气。
    """
    if not hits:
        return dedent(
            """
            <div class='ai-answer'>
              <div class='ai-answer-meta'>AI 智能助手</div>
              <p class='ai-answer-main'>抱歉，未检索到与您问题强相关的政策或文档。建议您精简关键词或尝试询问其他业务。</p>
            </div>
            """
        ).strip()

    top_hit = hits[0]
    main_text = top_hit.get("pure_text") or top_hit.get("snippet") or top_hit.get("content", "")
    main_text = smart_truncate(clean_snippet_text(main_text), 250)

    refs_html = []
    for i, h in enumerate(hits[:3], start=1):
        refs_html.append(f"<span class='ai-chip'>引用[{i}] {html.escape(h['title'])}</span>")
    refs_section = "".join(refs_html)

    return dedent(
        f"""
        <div class='ai-answer'>
          <div class='ai-answer-meta'>AI 智能总结</div>
          <div class='ai-answer-question'>{html.escape(query)}</div>
          <p class='ai-answer-main'>{html.escape(main_text)}</p>

          <div class='ai-answer-chips'>
             {refs_section}
          </div>

          <div class='ai-answer-foot'>
            已根据检索结果为您生成回答，详细出处请见下方“引用来源”。
          </div>
        </div>
        """
    ).strip()


# 将日期转换为友好的截止时间描述
def pretty_deadline(deadline: str | None) -> str:
    if not deadline:
        return "长期有效"
    try:
        target = dtparse(deadline).date()
    except Exception:
        return deadline
    today = datetime.now().date()
    diff = (target - today).days
    if diff < 0:
        return f"已截止({target:%Y-%m-%d})"
    if diff == 0:
        return "今日截止"
    if diff == 1:
        return "明日截止"
    return f"{target:%Y-%m-%d} · 剩余 {diff} 天"


# 对通知文本做兴趣匹配
def similarity_match(texts: list[str], keywords: list[str]):
    if not texts:
        return np.array([], dtype=int), np.array([])
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=False,
        max_features=2000,
    )
    matrix = vectorizer.fit_transform(texts)
    query = " ".join(filter(None, keywords)) or "通知 公告 校园"
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    max_score = float(sims.max()) if sims.size else 0.0
    if max_score > 0:
        normalized = sims / max_score
    else:
        normalized = sims
    normalized = np.clip(normalized, 0.0, 1.0)
    order = np.argsort(-normalized)
    return order, normalized


# 统计最热问题与失败问题
def compute_insights(log: pd.DataFrame):
    hot = (
        log.groupby("q")
        .agg(freq=("hits", "sum"), fails=("ok", lambda x: (~x).sum()))
        .reset_index()
        .sort_values("freq", ascending=False)
        .head(10)
    )
    fail = (
        hot[hot["fails"] > 0][["q", "fails"]]
        .rename(columns={"fails": "fails"})
        .sort_values("fails", ascending=False)
        .head(5)
    )
    return hot[["q", "freq"]], fail


# 根据失败问题生成后续动作建议
def recommend_actions(top_fail: pd.DataFrame) -> list[str]:
    actions: list[str] = []
    for _, row in top_fail.iterrows():
        question = row["q"]
        actions.append(f"为“{question}”补录流程指引, 并在知识库中添加答疑。")
    if not actions:
        actions.append("持续监控查询日志, 保持知识库与流程的月度复盘迭代。")
    return actions


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid hex color: {color}")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _gradient_palette(start: str, end: str, steps: int) -> list[str]:
    if steps <= 1:
        return [start]
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    denom = max(steps - 1, 1)
    palette: list[str] = []
    for idx in range(steps):
        ratio = idx / denom
        rgb = [
            round(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * ratio)
            for i in range(3)
        ]
        palette.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
    return palette


def style_bar_chart(
    fig,
    start_color: str = "#5C6CFF",
    end_color: str = "#8CD5C1",
    *,
    x_title: str = "",
    y_title: str = "",
    hover_template: str | None = None,
):
    """统一 Plotly 柱状图风格，形成与 UI 一致的玻璃拟态视觉。"""
    if not fig.data:
        return fig

    trace = fig.data[0]

    def _ensure_sequence(values):
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return list(values)
        try:
            return list(values)
        except TypeError:
            return [values]

    x_values = _ensure_sequence(getattr(trace, "x", None))
    y_values = _ensure_sequence(getattr(trace, "y", None))
    steps = len(x_values) or len(y_values) or 1
    palette = _gradient_palette(start_color, end_color, steps)
    fig.update_traces(
        marker=dict(
            color=palette,
            line=dict(color="rgba(255,255,255,0.65)", width=1.4),
            opacity=0.95,
        ),
        hovertemplate=hover_template or "<b>%{x}</b><br>%{y}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_tickangle=-32,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, Inter, sans-serif", color="#0F172A", size=13),
        margin=dict(l=30, r=10, t=60, b=65),
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.9)",
            font_size=13,
            font_family="Manrope, Inter, sans-serif",
        ),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12, color="#475467"))
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.25)",
        zeroline=False,
        tickfont=dict(size=12, color="#475467"),
    )
    return fig


VEC, MAT = build_tfidf_corpus(DOCS)
# -------------------- 页面配置与样式 --------------------


# 初始化 Streamlit 页面配置
def set_page_config():
    if st.session_state.get("_page_configured"):
        return
    st.set_page_config(
        page_title="RUC Connect · Executive Intelligence Cockpit",
        page_icon="🎓",
        layout="wide",
    )
    st.session_state["_page_configured"] = True


# 注入全局 CSS, 构建品牌化界面
def inject_global_styles():
    css = dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

        :root {
          --brand-700: #2329FF;
          --brand-600: #2F44FF;
          --brand-500: #4E63FF;
          --brand-400: #7A8CFF;
          --accent-emerald: #18C7B4;
          --accent-sun: #FFB678;
          --surface: rgba(255,255,255,0.82);
          --surface-alt: rgba(255,255,255,0.68);
          --surface-deep: rgba(255,255,255,0.55);
          --glass-border: rgba(255,255,255,0.32);
          --glass-border-strong: rgba(124,141,255,0.8);
          --text-strong: #0F172A;
          --text-muted: #5C6280;
          --radius-xl: 32px;
          --radius-lg: 26px;
          --radius-md: 18px;
          --shadow-lg: 0 32px 85px rgba(15, 23, 42, 0.2);
          --shadow-md: 0 24px 54px rgba(15, 23, 42, 0.16);
          --shadow-soft: 0 18px 44px rgba(15, 23, 42, 0.09);
          --blur-strong: 28px;
        }

        html, body, [class*="css"] {
          font-family: 'Manrope', 'Space Grotesk', 'Inter', 'Microsoft YaHei', 'PingFang SC', sans-serif;
          color: var(--text-strong);
          background: transparent;
          line-height: 1.65;
          letter-spacing: 0.01em;
          text-rendering: optimizeLegibility;
          -webkit-font-smoothing: antialiased;
        }

        .stApp {
          background:
            radial-gradient(circle at 8% 12%, rgba(82, 91, 255, 0.18), transparent 52%),
            radial-gradient(circle at 86% 2%, rgba(18, 199, 197, 0.18), transparent 42%),
            radial-gradient(circle at 35% 85%, rgba(255, 182, 116, 0.17), transparent 46%),
            linear-gradient(180deg, #EEF2FF 0%, #F7F4FF 42%, #FCF2FF 100%);
          position: relative;
          min-height: 100vh;
          overflow-x: hidden;
        }

        .stApp::before,
        .stApp::after {
          content: "";
          position: fixed;
          width: 540px;
          height: 540px;
          border-radius: 50%;
          filter: blur(210px);
          z-index: -1;
          opacity: 0.5;
          pointer-events: none;
        }

        .stApp::before {
          background: rgba(79,70,229,0.25);
          top: -200px;
          left: -150px;
        }

        .stApp::after {
          background: rgba(16,185,129,0.25);
          bottom: -180px;
          right: -180px;
        }

        .block-container {
          padding: 3.2rem 3.2rem 3.4rem;
          max-width: 1240px;
        }

        .stApp p {
          margin-bottom: 0.45rem;
        }

        .fade-in {
          opacity: 0;
          transform: translateY(18px);
          animation: floatUp .65s ease forwards;
        }

        @keyframes floatUp {
          to { opacity: 1; transform: translateY(0); }
        }

        .stButton>button {
          width: 100%;
          border-radius: 999px;
          background: linear-gradient(120deg, rgba(79,70,229,0.95), rgba(67,112,255,0.9) 60%, rgba(67,217,173,0.85));
          border: 1px solid var(--glass-border);
          color: #fff;
          padding: 0.78rem 1.4rem;
          font-weight: 600;
          font-size: .95rem;
          box-shadow: 0 22px 42px rgba(65, 91, 255, 0.32);
          backdrop-filter: blur(14px) saturate(120%);
          transition: transform .22s ease, box-shadow .25s ease, letter-spacing .25s ease;
        }

        .stButton>button:hover {
          transform: translateY(-3px) scale(1.01);
          letter-spacing: 0.02em;
          box-shadow: 0 30px 55px rgba(65, 91, 255, 0.38);
        }

        .stButton>button:focus:not(:active) {
          box-shadow: 0 0 0 3px rgba(91,115,255,0.35);
        }

        .stButton>button:disabled {
          opacity: .6;
          cursor: not-allowed;
        }

        .hero {
          border-radius: var(--radius-xl);
          padding: 42px 52px;
          background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(240,244,255,0.82));
          border: 1px solid var(--glass-border);
          box-shadow: var(--shadow-lg);
          position: relative;
          overflow: hidden;
          backdrop-filter: blur(var(--blur-strong));
        }

        .hero::before,
        .hero::after {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
        }

        .hero::before {
          background: radial-gradient(circle at 12% -5%, rgba(58,91,255,0.25), transparent 65%);
        }

        .hero::after {
          background: radial-gradient(circle at 82% 18%, rgba(18,184,134,0.2), transparent 55%);
        }

        .hero-pill {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.35rem 0.85rem;
          border-radius: 999px;
          text-transform: uppercase;
          font-size: .75rem;
          letter-spacing: 0.14em;
          background: rgba(255,255,255,0.45);
          border: 1px solid rgba(255,255,255,0.5);
          color: var(--brand-500);
          position: relative;
          z-index: 1;
        }

        .hero h1 {
          font-size: 2.45rem;
          margin: 0.75rem 0 0.25rem;
          position: relative;
          z-index: 1;
          letter-spacing: -0.02em;
        }

        .hero-subtitle {
          margin: 0;
          font-size: 1.1rem;
          color: var(--text-muted);
          position: relative;
          z-index: 1;
        }

        .hero-foot {
          margin-top: 1.4rem;
          display: flex;
          gap: 1rem;
          flex-wrap: wrap;
          font-size: .92rem;
          color: var(--text-muted);
          position: relative;
          z-index: 1;
        }

        .hero-foot span {
          padding: 0.35rem 0.9rem;
          border-radius: 999px;
          background: rgba(255,255,255,0.55);
          border: 1px solid rgba(255,255,255,0.4);
        }

        .section-header {
          margin: 2.8rem 0 1.45rem;
        }

        .section-header h2 {
          margin: 0;
          font-size: 1.65rem;
        }

        .section-header p {
          margin: .4rem 0 0;
          color: var(--text-muted);
        }

        .pill {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          font-size: .75rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: .08em;
          padding: 0.35rem 0.7rem;
          border-radius: 999px;
          background: rgba(91,115,255,0.16);
          color: var(--brand-500);
          border: 1px solid rgba(91,115,255,0.3);
        }

        .metric-card,
        .value-card,
        .showcase-card,
        .callout-card,
        .source-card,
        .notification-card,
        .answer-box {
          border-radius: var(--radius-lg);
          background: var(--surface);
          border: 1px solid var(--glass-border);
          box-shadow: var(--shadow-soft);
          position: relative;
          overflow: hidden;
          backdrop-filter: blur(18px) saturate(130%);
          transition: transform .25s ease, box-shadow .3s ease, border-color .3s ease;
        }

        .metric-card::after,
        .value-card::after,
        .callout-card::after,
        .showcase-card::after,
        .source-card::after,
        .notification-card::after,
        .answer-box::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(120deg, rgba(255,255,255,0.15), transparent 55%);
          opacity: 0;
          pointer-events: none;
          transition: opacity .3s ease;
        }

        .metric-card:hover,
        .value-card:hover,
        .showcase-card:hover,
        .callout-card:hover,
        .source-card:hover,
        .notification-card:hover,
        .answer-box:hover {
          transform: translateY(-6px);
          box-shadow: var(--shadow-md);
          border-color: var(--glass-border-strong);
        }

        .metric-card:hover::after,
        .value-card:hover::after,
        .showcase-card:hover::after,
        .callout-card:hover::after,
        .source-card:hover::after,
        .notification-card:hover::after,
        .answer-box:hover::after {
          opacity: 1;
        }

        .metric-card {
          padding: 24px 26px;
        }

        .metric-card .value {
          font-size: 1.9rem;
          font-weight: 700;
          margin-top: .2rem;
          color: var(--text-strong);
        }

        .metric-card .desc {
          margin-top: .55rem;
          font-size: .93rem;
          color: var(--text-muted);
        }

        .metric-card .badge {
          position: absolute;
          top: 18px;
          right: 22px;
          font-size: .78rem;
          padding: 0.28rem 0.75rem;
          border-radius: 999px;
          background: rgba(58,91,255,0.16);
          color: var(--brand-500);
          border: 1px solid rgba(58,91,255,0.3);
        }

        .value-card {
          padding: 26px 30px;
        }

        .value-card h4 {
          margin: 0 0 .65rem;
          font-size: 1.08rem;
        }

        .value-card p {
          margin: 0;
          color: var(--text-muted);
          line-height: 1.58;
        }

        .showcase-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.4rem;
          margin: 1.4rem 0 2.2rem;
        }

        .showcase-card {
          padding: 28px 30px;
          background: linear-gradient(150deg, rgba(255,255,255,0.92), rgba(243,246,255,0.75));
        }

        .showcase-card::before {
          content: "";
          position: absolute;
          inset: -30% 40% 35% -15%;
          background: var(--accent-grad, rgba(58,91,255,0.16));
          filter: blur(42px);
        }

        .showcase-icon {
          width: 46px;
          height: 46px;
          border-radius: 16px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 22px;
          color: #fff;
          margin-bottom: 22px;
          position: relative;
          z-index: 1;
          box-shadow: 0 18px 36px rgba(0,0,0,0.08);
        }

        .quick-actions {
          margin-top: 2.6rem;
          margin-bottom: 1rem;
        }

        .quick-action-caption,
        .callout-card li,
        .callout-card p,
        .notification-meta,
        .source-meta {
          color: var(--text-muted);
        }

        .quick-action-caption {
          margin-top: .65rem;
          font-size: .88rem;
          text-align: center;
        }

        .callout-card {
          padding: 24px 26px;
          background: linear-gradient(160deg, rgba(255,255,255,0.94), rgba(245,248,255,0.84));
        }

        .callout-card h4 {
          margin: 0 0 .8rem;
        }

        .callout-card ul {
          margin: 0;
          padding-left: 1.2rem;
        }

        .callout-card li {
          margin-bottom: .45rem;
          line-height: 1.55;
        }

        .answer-box {
          padding: 28px 30px;
          line-height: 1.65;
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          background:
            linear-gradient(125deg, rgba(75,92,255,0.18), rgba(255,255,255,0.88)),
            rgba(255,255,255,0.5);
          box-shadow: 0 35px 85px rgba(79,70,229,0.18);
          backdrop-filter: blur(28px) saturate(140%);
          position: relative;
          overflow: hidden;
        }

        .answer-box::before,
        .answer-box::after {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
        }

        .answer-box::before {
          background: radial-gradient(circle at 10% 10%, rgba(255,255,255,0.55), transparent 60%);
          opacity: 0.8;
        }

        .answer-box::after {
          background: linear-gradient(140deg, rgba(255,255,255,0.08), transparent 55%);
        }

        .ai-answer {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          position: relative;
          z-index: 1;
        }

        .ai-answer-meta {
          font-size: .78rem;
          letter-spacing: 0.2em;
          text-transform: uppercase;
          color: rgba(15,23,42,0.65);
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
        }

        .ai-answer-meta::before {
          content: "✨";
          width: 32px;
          height: 32px;
          border-radius: 12px;
          background: rgba(255,255,255,0.8);
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 1rem;
          color: var(--brand-600);
          box-shadow: 0 10px 25px rgba(79,70,229,0.25);
        }

        .ai-answer-question {
          font-size: 1.18rem;
          font-weight: 600;
          color: var(--text-strong);
        }

        .ai-answer-main {
          margin: 0;
          font-size: 1rem;
          color: var(--text-strong);
          line-height: 1.8;
        }

        .ai-answer-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem;
        }

        .ai-chip {
          padding: 0.4rem 0.85rem;
          border-radius: 999px;
          border: 1px solid rgba(91,115,255,0.25);
          background: linear-gradient(120deg, rgba(255,255,255,0.95), rgba(209,213,255,0.5));
          font-size: .82rem;
          color: var(--brand-500);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.7);
        }

        .ai-answer-foot {
          font-size: .82rem;
          color: rgba(15,23,42,0.65);
          border-top: 1px solid rgba(255,255,255,0.4);
          padding-top: 0.6rem;
        }

        .reference-stack {
          margin-top: 0.8rem;
          display: flex;
          flex-direction: column;
          gap: 1.1rem;
          position: relative;
          padding-left: 0.2rem;
        }

        .reference-divider {
          margin: 1.4rem 0 0.4rem;
          display: flex;
          align-items: center;
          gap: 0.75rem;
          font-size: 0.85rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: rgba(15,23,42,0.65);
        }

        .reference-divider::after {
          content: "";
          flex: 1;
          height: 1px;
          background: linear-gradient(90deg, rgba(91,115,255,0.35), rgba(91,115,255,0));
        }

        .reference-stack::before {
          content: "";
          position: absolute;
          left: 22px;
          top: 10px;
          bottom: 10px;
          width: 2px;
          background: linear-gradient(180deg, rgba(91,115,255,0.25), rgba(91,115,255,0));
        }

        .reference-stack > * {
          position: relative;
          z-index: 1;
        }

        .source-card {
          padding: 22px 24px;
          margin-top: 0.6rem;
          display: flex;
          gap: 1rem;
          align-items: flex-start;
          position: relative;
          border: 1px solid rgba(91,115,255,0.18);
          border-radius: var(--radius-lg);
          background: linear-gradient(150deg, rgba(255,255,255,0.92), rgba(245,248,255,0.78));
          box-shadow: 0 25px 55px rgba(15,23,42,0.08);
          backdrop-filter: blur(18px);
        }

        .source-card::after {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: inherit;
          border: 1px solid rgba(255,255,255,0.35);
          pointer-events: none;
        }

        .source-card .source-body {
          flex: 1;
          position: relative;
          z-index: 1;
        }

        .source-index {
          width: 46px;
          height: 46px;
          border-radius: 16px;
          background: linear-gradient(135deg, #5B66FF, #8ADFD2);
          border: 1px solid rgba(255,255,255,0.65);
          color: #fff;
          font-weight: 700;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: .95rem;
          flex-shrink: 0;
          box-shadow: 0 18px 35px rgba(50,65,197,0.25);
        }

        .notification-card {
          padding: 24px 26px;
        }

        .notification-card h4 {
          margin: 0 0 .5rem;
        }

        .notification-card a {
          color: var(--brand-500);
          font-weight: 600;
        }

        section[data-testid="stSidebar"] > div {
          background: rgba(255,255,255,0.75);
          border-right: 1px solid rgba(255,255,255,0.35);
          box-shadow: inset -1px 0 0 rgba(255,255,255,0.35);
          backdrop-filter: blur(22px);
        }

        .sidebar-brand h2 {
          margin-bottom: .2rem;
        }

        .sidebar-brand p {
          color: var(--text-muted);
          font-size: .9rem;
        }

        .sidebar-metric {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: .85rem;
        }

        .sidebar-metric span {
          color: var(--text-muted);
          font-size: .85rem;
        }

        .sidebar-metric strong {
          font-size: 1.05rem;
          color: var(--text-strong);
        }

        .stTabs [data-baseweb="tab"] {
          font-weight: 600;
          color: var(--text-muted);
          border-radius: 999px;
          padding: 0.4rem 1.1rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
          color: var(--brand-500);
          background: rgba(58,91,255,0.08);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
          color: var(--brand-500);
          background: rgba(58,91,255,0.14);
        }

        .stTabs [data-baseweb="tab-highlight"] {
          background: transparent;
        }

        .stPlotlyChart {
          padding: 1.2rem;
          border-radius: var(--radius-lg);
          background: var(--surface);
          border: 1px solid var(--glass-border);
          box-shadow: var(--shadow-soft);
          backdrop-filter: blur(18px);
        }

        .stTextInput>div>div>input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {
          border-radius: 16px;
          border: 1px solid var(--glass-border);
          background: var(--surface-deep);
          padding: 0.65rem 0.9rem;
          backdrop-filter: blur(18px);
          transition: border-color .2s ease, box-shadow .2s ease;
        }

        .stTextInput>div>div>input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus,
        .stSelectbox div[data-baseweb="select"]:focus,
        .stMultiSelect div[data-baseweb="select"]:focus {
          border-color: var(--glass-border-strong);
          box-shadow: 0 0 0 2px rgba(91,115,255,0.25);
        }

        @media (max-width: 1024px) {
          .block-container {
            padding: 2.4rem 1.7rem 2.6rem;
          }

          .hero {
            padding: 32px;
          }

          .hero-foot span {
            width: 100%;
          }

          .showcase-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """
    )
    st.markdown(css, unsafe_allow_html=True)
    st.session_state["_css_signature"] = hashlib.md5(css.encode("utf-8")).hexdigest()
# -------------------- UI 组件 --------------------


# 渲染区段标题与副标题
def render_section_header(title: str, subtitle: str, badge: str | None = None, delay: float = 0.0):
    badge_html = f"<span class='pill'>{html.escape(badge)}</span>" if badge else ""
    st.markdown(
        dedent(
            f"""
<div class='section-header fade-in' style='animation-delay:{delay:.2f}s;'>
  {badge_html}
  <h2>{html.escape(title)}</h2>
  <p>{html.escape(subtitle)}</p>
</div>
            """
        ),
        unsafe_allow_html=True,
    )


# 渲染 KPI 卡片
def render_metric_card(title: str, value: str, description: str, badge: str | None = None, delay: float = 0.0):
    badge_html = f"<div class='badge'>{badge}</div>" if badge else ""
    st.markdown(
        dedent(
            f"""
<div class='metric-card fade-in' style='animation-delay:{delay:.2f}s;'>
  {badge_html}
  <h3>{html.escape(title)}</h3>
  <div class='value'>{html.escape(value)}</div>
  <div class='desc'>{html.escape(description)}</div>
</div>
            """
        ),
        unsafe_allow_html=True,
    )


# 渲染价值描述卡片
def render_value_card(title: str, description: str, delay: float = 0.0):
    st.markdown(
        dedent(
            f"""
<div class='value-card fade-in' style='animation-delay:{delay:.2f}s;'>
  <h4>{html.escape(title)}</h4>
  <p>{html.escape(description)}</p>
</div>
            """
        ),
        unsafe_allow_html=True,
    )


# 以网格展示亮点卡片
def render_showcase_grid(items: list[dict[str, str]]):
    cards = []
    for item in items:
        icon_bg = item.get("icon_bg", "linear-gradient(135deg,#3A5BFF,#5B73FF)")
        icon_shadow = item.get("icon_shadow", "rgba(58,91,255,0.35)")
        icon = item.get("icon", "📌")
        title = html.escape(item.get("title", ""))
        desc = html.escape(item.get("description", ""))
        cards.append(
            dedent(
                f"""
<div class='showcase-card' style='--accent-grad:{item.get("accent", "rgba(58,91,255,0.18)")};'>
  <div class='showcase-icon' style='background:{icon_bg}; box-shadow: 0 18px 32px {icon_shadow};'>{icon}</div>
  <h4>{title}</h4>
  <p>{desc}</p>
</div>
                """
            ).strip()
        )
    st.markdown(
        dedent(
            f"""
<div class='showcase-grid'>
  {''.join(cards)}
</div>
            """
        ),
        unsafe_allow_html=True,
    )


# 渲染首页快捷导航按钮
def render_quick_actions() -> str | None:
    st.markdown(
        dedent(
            """
<div class='quick-actions fade-in' style='animation-delay: 0.1s;'>
  <h3>快速体验</h3>
  <p>一键切换核心能力, 感受端到端的智能工作流</p>
</div>
            """
        ),
        unsafe_allow_html=True,
    )

    actions = [
        ("🧠", "智能问答 RAG", "检索权威文档, 带引用作答", "rag"),
        ("🤖", "个性化 Agent", "画像驱动的主动推送", "agent"),
        ("📊", "服务洞察 Admin", "热点盲点实时监控", "admin"),
        ("📘", "Executive Overview", "回到驾驶舱总览", "home"),
    ]

    cols = st.columns(len(actions))
    selected = None
    for idx, (icon, label, caption, page_key) in enumerate(actions):
        with cols[idx]:
            if st.button(f"{icon} {label}", key=f"quick_nav_{page_key}", use_container_width=True):
                selected = page_key
            st.markdown(
                f"<p class='quick-action-caption'>{html.escape(caption)}</p>",
                unsafe_allow_html=True,
            )
    return selected


NAV_ITEMS = [
    ("Executive Overview", "home"),
    ("Intelligent QA (RAG)", "rag"),
    ("Personalized Assistant", "agent"),
    ("Service Intelligence", "admin"),
]

NAV_DICT = {label: key for label, key in NAV_ITEMS}
NAV_REVERSE_DICT = {key: label for label, key in NAV_ITEMS}


# 统计平台指标概览
def calculate_platform_metrics():
    total_docs = len(DOCS)
    total_notices = len(NOTICES)
    total_queries = QUERY_LOG.shape[0]
    success_ratio = float(QUERY_LOG["ok"].mean()) if total_queries else 0.0
    blindspots = int((~QUERY_LOG["ok"]).sum())
    avg_hits = float(QUERY_LOG["hits"].mean()) if total_queries else 0.0
    return {
        "total_docs": total_docs,
        "total_notices": total_notices,
        "total_queries": total_queries,
        "success_ratio": success_ratio,
        "blindspots": blindspots,
        "avg_hits": avg_hits,
    }


# 渲染侧边栏导航与指标
def render_sidebar(metrics):
    with st.sidebar:
        st.markdown(
            dedent(
                """
<div class='sidebar-brand'>
  <h2>RUC Connect</h2>
  <p>Executive cockpit for data-informed campus governance.</p>
</div>
                """
            ),
            unsafe_allow_html=True,
        )

        nav_labels = [item[0] for item in NAV_ITEMS]
        current_label = NAV_REVERSE_DICT.get(st.session_state.get("page", "home"), nav_labels[0])
        selected = st.radio(
            "导航",
            nav_labels,
            index=nav_labels.index(current_label),
            label_visibility="collapsed",
        )
        st.session_state["page"] = NAV_DICT[selected]

        st.markdown("---")
        st.markdown("**Platform pulse**")
        success_rate = f"{metrics['success_ratio'] * 100:.0f}%"
        blindspots = metrics["blindspots"]
        st.markdown(
            dedent(
                """
<div class='sidebar-metric'>
  <span>成功率</span>
  <strong>{success_rate}</strong>
</div>
<div class='sidebar-metric'>
  <span>盲点待补全</span>
  <strong>{blindspots}</strong>
</div>
                """
            ).format(success_rate=success_rate, blindspots=blindspots),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("Crafted by AI Campus Innovation Lab · Ready for enterprise deployment")
# -------------------- 页面渲染逻辑 --------------------


# 首页头部英雄区
def render_hero_section(metrics):
    st.markdown(
        dedent(
            f"""
<div class='hero'>
  <div class='hero-pill'>Glass cockpit · Neo gradients</div>
  <h1>{html.escape(APP_TITLE)}</h1>
  <p class='hero-subtitle'>{html.escape(APP_SUBTITLE)}</p>
  <div class='hero-foot'>
    <span>{html.escape(APP_TAGLINE)}</span>
    <span>LLM / RAG / Agents</span>
  </div>
</div>
            """
        ),
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card(
            "权威文档库",
            f"{metrics['total_docs']:02d}",
            "财务 / 科研 / 后勤跨域指引一体化",
            delay=0.15,
        )
    with metric_cols[1]:
        render_metric_card(
            "动态通知流",
            f"{metrics['total_notices']:02d}",
            "实时同步门户与部门公告数据",
            delay=0.25,
        )
    with metric_cols[2]:
        render_metric_card(
            "问答成功率",
            f"{metrics['success_ratio'] * 100:.0f}%",
            f"平均召回 {metrics['avg_hits']:.1f} 个权威引用",
            delay=0.35,
    )


# 首页主体布局
def render_home(metrics):
    render_hero_section(metrics)
    selected_page = render_quick_actions()

    render_section_header(
        "战略价值焦点",
        "三大模块闭环支撑跨国高校的智能治理",
        "Value drivers",
        delay=0.45,
    )

    showcase_items = [
        {
            "icon": "🧠",
            "title": "LLM + RAG 智能问答",
            "description": "深度检索权威文档, 自动拼接引用链路, 助力合规决策。",
            "accent": "rgba(58,91,255,0.22)",
            "icon_bg": "linear-gradient(140deg,#3A5BFF,#6B8CFF)",
            "icon_shadow": "rgba(58,91,255,0.30)",
        },
        {
            "icon": "🤝",
            "title": "跨部门知识协同",
            "description": "财务 / 科研 / 后勤数据一体化维护, 构建全球校园的知识中台。",
            "accent": "rgba(18,184,134,0.20)",
            "icon_bg": "linear-gradient(140deg,#12B886,#34D399)",
            "icon_shadow": "rgba(18,184,134,0.26)",
        },
        {
            "icon": "⚡",
            "title": "主动触发工作流",
            "description": "事件驱动推送审批、报销、预约, 嵌入学校现有流程体系。",
            "accent": "rgba(249,115,22,0.22)",
            "icon_bg": "linear-gradient(150deg,#F97316,#FACC15)",
            "icon_shadow": "rgba(249,115,22,0.28)",
        },
        {
            "icon": "🌐",
            "title": "多语言 & 全球部署",
            "description": "支持中英文双语界面, 兼容 SSO / IAM 与多云混合架构。",
            "accent": "rgba(59,130,246,0.22)",
            "icon_bg": "linear-gradient(150deg,#3B82F6,#60A5FA)",
            "icon_shadow": "rgba(59,130,246,0.28)",
        },
    ]
    render_showcase_grid(showcase_items)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_value_card(
            "企业级安全基线",
            "零信任网络、关键操作审计、内容安全过滤, 实现端到端可控。",
            delay=0.55,
        )
    with col2:
        render_value_card(
            "体验即 ROI",
            "标准流程模板 + KPI 监控, 即刻衡量上线成效, 提升服务感知。",
            delay=0.65,
        )
    with col3:
        render_value_card(
            "无限扩展场景",
            "招生、校友、供应链等场景可扩展, 打造一体多端的协同网络。",
            delay=0.75,
        )

    tabs = st.tabs(["体验路径", "安全合规", "未来演进"])
    with tabs[0]:
        st.markdown(
            dedent(
                """
- 国际校园多语言支持: 英文 / 中文界面随选, 保障跨国师生快速上手。
- 无缝集成: 对接校园门户 / IAM / 流程引擎, 兼容 SSO 与审计。
- 私有化部署: 提供 Docker / K8s 模板, 适配主数据中心或混合云架构。
                """
            )
        )
    with tabs[1]:
        st.markdown(
            dedent(
                """
- 数据权限: 按部门 / 角色配置检索边界, 敏感数据自动脱敏。
- 审计闭环: 完整的查询日志、引用链路与回答版本留痕, 满足合规。
- 模型安全: 专用安全网关 + 本地审计策略, 确保生成内容可信可控。
                """
            )
        )
    with tabs[2]:
        st.markdown(
            dedent(
                """
- 行业模型: 结合高校垂直语料持续微调, 沉淀治理知识图谱。
- 多代理协作: 学工 / 财务 / 科研等角色化 Agent 协同运行。
- 全渠道触达: 支持 Web / 小程序 / 邮件 / IM 多端一体化推送。
                """
            )
        )

    return selected_page


# RAG 页面: 智能问答演示
def render_rag(metrics):
    render_section_header(
        "智能问答(RAG)",
        "基于权威文件的深度检索与解答, 支持即时引用链路。",
        "Decision support",
        delay=0.1,
    )

    col_left, col_right = st.columns([1.8, 1])
    with col_left:
        query = st.text_input("请输入问题", value="最近有哪些教授发表了高质量的论文")

        with st.expander("检索参数设置 (高级)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                topk = st.slider("返回引用数量", 1, 5, 3, key="rag_topk")
            with c2:
                min_score = st.slider("最低相似度阈值", 0.0, 1.0, 0.15, 0.01, key="rag_minscore")

        if st.button("检索并生成回答", type="primary"):
            with st.spinner("正在检索权威资料并汇总要点…"):
                hits, low_confidence = rag_search(
                    query,
                    DOCS,
                    VEC,
                    MAT,
                    topk=topk,
                    min_score=min_score,
                )
                answer = synthesize_answer(query, hits)

            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

            if hits and low_confidence:
                st.warning("提示：检索结果相似度较低，以下内容仅供参考。")

            if hits:
                st.markdown(
                    "<div class='reference-divider'>引用来源 · Reference Sources</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='reference-stack'>", unsafe_allow_html=True)
                for idx, h in enumerate(hits, start=1):
                    body_text = h.get("pure_text") or h.get("snippet") or h["chunk"]
                    body_text = smart_truncate(body_text, 180)
                    match_info = ""
                    if h.get("source_query"):
                        match_info = (
                            f"<div style='font-size: 0.85rem; color: #64748B; margin-bottom:4px;'>匹配知识库问答: "
                            f"{html.escape(h['source_query'])}</div>"
                        )
                    st.markdown(
                        dedent(
                            f"""
<div class='source-card'>
  <div class='source-index'>{idx}</div>
  <div class='source-body'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <b style='font-size:1rem;'>{html.escape(h['title'])}</b>
        <span class='pill' style='font-size:0.7rem;'>相似度 {h['score']:.2f}</span>
    </div>
    <div class='source-meta' style='margin-top:2px; margin-bottom:8px;'>部门: {html.escape(h['dept'])}</div>
    {match_info}
    <div style='background:rgba(241,245,249,0.5); padding:8px; border-radius:8px; font-size:0.9rem; color:#334155;'>
        {html.escape(body_text)}
    </div>
    <div style='margin-top: 8px; text-align:right;'>
        <a href='{h['url']}' target='_blank' style='font-size:0.85rem; color:var(--brand-600); text-decoration:none;'>📄 查看原文文档 &rarr;</a>
    </div>
  </div>
</div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("未找到满足条件的引用来源。")

    with col_right:
        st.markdown(
            dedent(
                """
<div class='callout-card fade-in' style='animation-delay: 0.2s;'>
  <h4>RAG 能力亮点</h4>
  <ul>
    <li>知识粒度: 支持 PDF / Word / Excel / HTML 多格式解析</li>
    <li>多语种: 中英双语向量检索, 全球校区无缝复用</li>
    <li>审计追踪: 问题、召回、回答版本全链路留痕</li>
    <li>生态延展: 对接 CRM / ERP / 门户, 实现跨系统问答</li>
  </ul>
</div>
                """
            ),
            unsafe_allow_html=True,
        )


# Agent 页面: 个性化推送体验
def render_agent(metrics):
    render_section_header(
        "个性化助理(Agent)",
        "根据画像精准推送通知、审批与风险事项, 打造智能待办。",
        "Proactive services",
        delay=0.1,
    )

    with st.expander("配置画像", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            identity = st.selectbox("身份", ["本科生", "硕士研究生", "博士研究生", "青年教师"], index=2)
            school = st.selectbox(
                "学院",
                ["公共管理学院", "法学院", "经济学院", "信息学院", "智慧治理学院"],
                index=0,
            )
        with c2:
            interests = st.multiselect(
                "关注主题",
                [
                    "基金申报",
                    "数字治理",
                    "AIGC",
                    "合规",
                    "医保",
                    "打印",
                    "本科生课程",
                    "研究生课程",
                    "实验室预约",
                    "财务报销",
                ],
                default=["基金申报", "数字治理"],
            )
        with c3:
            current_need = st.text_input("当前诉求 / 待办说明", "想了解基金申报与经费报销的最新政策")

    with st.expander("推送参数 (高级)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            agent_topk = st.slider("推荐条数", 1, 5, 3, key="agent_topk")
        with col_b:
            agent_min_score = st.slider("最低匹配分", 0.0, 1.0, 0.18, 0.01, key="agent_min_score")

    if st.button("生成智能推送", type="primary"):
        persona_query = " ".join(filter(None, [identity, school, current_need, *interests]))
        if not persona_query.strip():
            persona_query = identity
        hits, low_confidence = rag_search(
            persona_query,
            DOCS,
            VEC,
            MAT,
            topk=agent_topk,
            min_score=agent_min_score,
        )
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.markdown("### 推荐卡片")
        if low_confidence and hits:
            st.info("当前匹配分偏低，以下推荐为相似度自动回退结果。")

        if hits:
            for idx, rec in enumerate(hits, start=1):
                delay = 0.12 * (idx - 1) + 0.15
                snippet = rec.get("pure_text") or rec.get("snippet") or rec["chunk"]
                question_html = ""
                if rec.get("source_query"):
                    question_html = (
                        f"<div class='notification-meta' style='margin-top:4px;'>知识库问法: {html.escape(rec['source_query'])}</div>"
                    )
                st.markdown(
                    dedent(
                        f"""
<div class='notification-card fade-in' style='animation-delay: {delay:.2f}s;'>
  <h4>{html.escape(rec['title'])}</h4>
  <div class='notification-meta'>
    来自: {html.escape(rec['dept'])} ｜ 匹配度: {rec['score']:.2f}
  </div>
  {question_html}
  <div style='margin-top: 10px'>{html.escape(snippet)}</div>
  <div style='margin-top: 12px'><a href='{rec['url']}' target='_blank'>访问相关网址</a></div>
</div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                dedent(
                    """
<div class='callout-card fade-in' style='animation-delay: 0.15s;'>
  <h4>暂无符合阈值的推送</h4>
  <p>尝试降低匹配分或补充更具体的诉求与兴趣词。</p>
</div>
                    """
                ),
                unsafe_allow_html=True,
            )


# Admin 页面: 服务洞察与图表
def render_admin(metrics):
    render_section_header(
        "服务洞察(Admin)",
        "以数据驱动的热点/盲点分析, 支撑知识库与流程的持续优化。",
        "Insight engine",
        delay=0.1,
    )

    total_queries = metrics["total_queries"]
    success_ratio = metrics["success_ratio"]
    blindspots = metrics["blindspots"]

    c1, c2, c3 = st.columns(3)
    c1.metric("总查询量", total_queries, f"成功率 {success_ratio * 100:.0f}%")
    c2.metric("盲点数量", blindspots, "待补全")
    c3.metric("平均召回文档", f"{metrics['avg_hits']:.1f}", "+/- 0.3 vs 上周")

    top_hot, top_fail = compute_insights(QUERY_LOG)

    left, right = st.columns(2)
    with left:
        fig_hot = px.bar(top_hot, x="q", y="freq", title="Top10 高频查询", text_auto=True)
        fig_hot = style_bar_chart(
            fig_hot,
            start_color="#5B66FF",
            end_color="#7BD7C5",
            x_title="问题",
            y_title="次数",
            hover_template="<b>%{x}</b><br>访问 %{y} 次<extra></extra>",
        )
        st.plotly_chart(fig_hot, use_container_width=True, config={"displayModeBar": False})

    with right:
        if not top_fail.empty:
            fig_fail = px.bar(top_fail, x="q", y="fails", title="Top5 查询失败", text_auto=True)
            fig_fail = style_bar_chart(
                fig_fail,
                start_color="#FF8A7A",
                end_color="#FFC3A2",
                x_title="问题",
                y_title="失败次数",
                hover_template="<b>%{x}</b><br>失败 %{y} 次<extra></extra>",
            )
            st.plotly_chart(fig_fail, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                dedent(
                    """
<div class='callout-card fade-in' style='animation-delay: 0.2s;'>
  <h4>暂无失败查询</h4>
  <p>继续保持知识库更新节奏, 关注新学期的场景需求。</p>
</div>
                    """
                ),
                unsafe_allow_html=True,
            )

    actions_html = "".join(
        f"<li><strong>{idx}.</strong> {html.escape(action)}</li>"
        for idx, action in enumerate(recommend_actions(top_fail), start=1)
    )
    st.markdown(
        dedent(
            f"""
<div class='callout-card fade-in' style='animation-delay: 0.3s;'>
  <h4>部门协同计划</h4>
  <ul style='list-style:none; padding-left:0;'>
    {actions_html}
  </ul>
</div>
            """
        ),
        unsafe_allow_html=True,
    )

# 运行入口
# 控制页面路由与状态管理
def build_app():
    set_page_config()
    inject_global_styles()

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    metrics = calculate_platform_metrics()
    render_sidebar(metrics)

    page = st.session_state.get("page", "home")
    if page == "home":
        target = render_home(metrics)
        if target and target != page:
            st.session_state["page"] = target
            rerun = getattr(st, "experimental_rerun", None)
            if callable(rerun):
                rerun()
                return
    elif page == "rag":
        render_rag(metrics)
    elif page == "agent":
        render_agent(metrics)
    elif page == "admin":
        render_admin(metrics)
    else:
        st.session_state["page"] = "home"
        render_home(metrics)


if __name__ == "__main__":
    build_app()
