from __future__ import annotations

import html
import hashlib
from datetime import datetime
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.parser import parse as dtparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 应用的标题与宣传文案
APP_TITLE = "RUC Connect · Executive Intelligence Cockpit"
APP_SUBTITLE = "AI驱动的一体化校园服务平台, 助力全球高校实现治理跃迁。"
APP_TAGLINE = "One campus. Every stakeholder. Decisions in minutes."


# 示例文档数据
DOCS = [
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
    },
]

# 示例通知公告
NOTICES = [
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

# 示例查询日志, 用于洞察统计
QUERY_LOG = pd.DataFrame(
    [
        {"q": "医保报销比例多少", "hits": 7, "ok": True},
        {"q": "打印店在哪", "hits": 4, "ok": True},
        {"q": "校车时刻表", "hits": 0, "ok": False},
        {"q": "核心期刊目录 A 类", "hits": 2, "ok": True},
        {"q": "补办校园卡怎么走流程", "hits": 5, "ok": False},
        {"q": "论文版面费报销标准", "hits": 2, "ok": True},
        {"q": "医保报销材料有哪些", "hits": 2, "ok": True},
        {"q": "打印彩色多少钱", "hits": 1, "ok": True},
        {"q": "研究生培养方案", "hits": 0, "ok": False},
        {"q": "基金申报截止时间", "hits": 1, "ok": True},
        {"q": "校医院挂号", "hits": 10, "ok": True},
        {"q": "体育馆开放时间", "hits": 0, "ok": False},
    ]
)
# -------------------- 核心工具函数 --------------------


# 将原始内容裁剪到指定长度, 保持展示整洁
def smart_truncate(text: str, max_len: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


# 构建 TF-IDF 向量用于检索
def build_tfidf_corpus(docs: list[dict[str, str]]):
    texts = [doc["content"] for doc in docs]
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# 基于向量相似度检索最相关的文档
def rag_search(query: str, docs: list[dict[str, str]], vectorizer, matrix, topk: int = 3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    indices = np.argsort(-sims)[:topk]
    results = []
    for idx in indices:
        doc = docs[idx]
        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "dept": doc["dept"],
                "url": doc["url"],
                "score": float(sims[idx]),
                "chunk": smart_truncate(doc["content"], 220),
                "content": doc["content"],
            }
        )
    return results


# 汇总检索结果, 生成易读回答
def synthesize_answer(query: str, hits: list[dict[str, str]]) -> str:
    if not hits:
        return "未检索到相关内容, 请尝试调整关键词或降低引用数量。"

    lead = smart_truncate(hits[0]["content"], 160)
    bullets = "\n".join(
        f"- **{hit['title']}** · {smart_truncate(hit['content'], 110)}" for hit in hits
    )
    return dedent(
        f"""
        **问题洞察**: {html.escape(query)}

        {lead}

        **引用摘要**

        {bullets}

        _基于 RAG 的权威文档检索与多源交叉验证_
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
    vectorizer = TfidfVectorizer(max_features=600)
    matrix = vectorizer.fit_transform(texts)
    query = " ".join(keywords) or "通知 公告 校园"
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    order = np.argsort(-sims)
    return order, sims


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
        actions.append(f"为“{question}”补录流程指引, 并在知识库中添加结构化答疑。")
    if not actions:
        actions.append("持续监控查询日志, 保持知识库与流程的月度复盘迭代。")
    return actions


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
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {
          --brand-600: #2F4CFF;
          --brand-500: #3A5BFF;
          --brand-400: #5B73FF;
          --accent-emerald: #12B886;
          --surface: rgba(255,255,255,0.92);
          --surface-alt: rgba(255,255,255,0.68);
          --border: rgba(15,23,42,0.08);
          --border-strong: rgba(15,23,42,0.14);
          --text-strong: #101828;
          --text-muted: #6B7280;
          --radius-lg: 24px;
          --radius-md: 18px;
          --shadow-lg: 0 28px 80px rgba(15, 23, 42, 0.08);
          --shadow-md: 0 22px 48px rgba(15, 23, 42, 0.07);
          --shadow-soft: 0 12px 30px rgba(15, 23, 42, 0.04);
        }

        html, body, [class*="css"] {
          font-family: 'Manrope', 'Inter', 'Microsoft YaHei', 'PingFang SC', sans-serif;
          color: var(--text-strong);
          background: transparent;
        }

        .stApp {
          background: linear-gradient(180deg, #EEF2FF 0%, #F6F8FF 55%, #FFFFFF 100%);
          position: relative;
          min-height: 100vh;
        }

        .stApp::before,
        .stApp::after {
          content: "";
          position: fixed;
          width: 520px;
          height: 520px;
          border-radius: 50%;
          filter: blur(180px);
          z-index: -1;
          opacity: 0.35;
          pointer-events: none;
        }

        .stApp::before {
          background: rgba(58,91,255,0.28);
          top: -180px;
          left: -140px;
        }

        .stApp::after {
          background: rgba(18,184,134,0.26);
          bottom: -180px;
          right: -160px;
        }

        .block-container {
          padding: 2.6rem 3.1rem 3rem;
          max-width: 1180px;
        }

        .fade-in {
          opacity: 0;
          transform: translateY(16px);
          animation: floatUp .6s ease forwards;
        }

        @keyframes floatUp {
          to { opacity: 1; transform: translateY(0); }
        }

        .stButton>button {
          width: 100%;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--brand-500), var(--brand-400));
          border: none;
          color: #fff;
          padding: 0.72rem 1.3rem;
          font-weight: 600;
          box-shadow: 0 16px 32px rgba(58,91,255,0.25);
          transition: transform .18s ease, box-shadow .2s ease;
        }

        .stButton>button:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 38px rgba(58,91,255,0.3);
        }

        .stButton>button:focus:not(:active) {
          box-shadow: 0 0 0 3px rgba(58,91,255,0.22);
        }

        .hero {
          border-radius: var(--radius-lg);
          padding: 38px 46px;
          background:
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(240,244,255,0.88)),
            var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-lg);
          position: relative;
          overflow: hidden;
        }

        .hero::before {
          content: "";
          position: absolute;
          inset: 0;
          background: radial-gradient(circle at 12% -10%, rgba(58,91,255,0.18), transparent 60%);
          pointer-events: none;
        }

        .hero::after {
          content: "";
          position: absolute;
          inset: 0;
          background: radial-gradient(circle at 85% 18%, rgba(18,184,134,0.16), transparent 55%);
          pointer-events: none;
        }

        .hero h1 {
          font-size: 2.3rem;
          margin: 0 0 .35rem 0;
          position: relative;
          z-index: 1;
        }

        .hero p {
          margin: 0;
          font-size: 1.05rem;
          color: var(--text-muted);
          position: relative;
          z-index: 1;
        }

        .section-header {
          margin: 2.4rem 0 1.4rem;
        }

        .section-header h2 {
          margin: 0;
          font-size: 1.6rem;
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
          background: rgba(58,91,255,0.12);
          color: var(--brand-500);
        }

        .metric-card,
        .value-card,
        .showcase-card,
        .callout-card,
        .source-card,
        .notification-card,
        .answer-box {
          border-radius: var(--radius-md);
          background: var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-soft);
          transition: transform .2s ease, box-shadow .25s ease, border-color .25s ease;
        }

        .metric-card:hover,
        .value-card:hover,
        .showcase-card:hover,
        .callout-card:hover,
        .source-card:hover,
        .notification-card:hover,
        .answer-box:hover {
          transform: translateY(-4px);
          box-shadow: var(--shadow-md);
          border-color: var(--border-strong);
        }

        .metric-card {
          padding: 22px 24px;
          position: relative;
        }

        .metric-card .value {
          font-size: 1.8rem;
          font-weight: 700;
          margin-top: .2rem;
          color: var(--text-strong);
        }

        .metric-card .desc {
          margin-top: .55rem;
          font-size: .92rem;
          color: var(--text-muted);
        }

        .metric-card .badge {
          position: absolute;
          top: 18px;
          right: 22px;
          font-size: .78rem;
          padding: 0.25rem 0.7rem;
          border-radius: 999px;
          background: rgba(58,91,255,0.14);
          color: var(--brand-500);
        }

        .value-card {
          padding: 24px 26px;
        }

        .value-card h4 {
          margin: 0 0 .6rem;
          font-size: 1.05rem;
        }

        .value-card p {
          margin: 0;
          color: var(--text-muted);
          line-height: 1.55;
        }

        .showcase-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.2rem;
          margin: 1.4rem 0 2.2rem;
        }

        .showcase-card {
          padding: 26px 28px;
          position: relative;
          overflow: hidden;
          background: linear-gradient(150deg, rgba(255,255,255,0.92), rgba(243,246,255,0.86));
        }

        .showcase-card::before {
          content: "";
          position: absolute;
          inset: -30% 40% 35% -10%;
          background: var(--accent-grad, rgba(58,91,255,0.16));
          filter: blur(40px);
        }

        .showcase-icon {
          width: 44px;
          height: 44px;
          border-radius: 14px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 21px;
          color: #fff;
          margin-bottom: 20px;
          position: relative;
          z-index: 1;
        }

        .showcase-card h4,
        .showcase-card p {
          position: relative;
          z-index: 1;
        }

        .showcase-card h4 {
          margin: 0 0 .55rem;
          font-size: 1.06rem;
        }

        .showcase-card p {
          margin: 0;
          color: var(--text-muted);
          font-size: .92rem;
          line-height: 1.55;
        }

        .quick-actions {
          margin-top: 2.5rem;
          margin-bottom: 1rem;
        }

        .quick-actions h3 {
          margin: 0 0 .3rem;
        }

        .quick-actions p,
        .quick-action-caption,
        .callout-card li,
        .callout-card p,
        .notification-meta,
        .source-meta {
          color: var(--text-muted);
        }

        .quick-action-caption {
          margin-top: .6rem;
          font-size: .88rem;
          text-align: center;
        }

        .callout-card {
          padding: 22px 24px;
          background: linear-gradient(160deg, rgba(255,255,255,0.94), rgba(245,248,255,0.88));
        }

        .callout-card h4 {
          margin: 0 0 .8rem;
        }

        .callout-card ul {
          margin: 0;
          padding-left: 1.1rem;
        }

        .callout-card li {
          margin-bottom: .45rem;
          line-height: 1.55;
        }

        .answer-box {
          padding: 22px 24px;
          line-height: 1.6;
        }

        .source-card {
          padding: 20px 22px;
          margin-top: 1rem;
        }

        .notification-card {
          padding: 22px 24px;
        }

        .notification-card h4 {
          margin: 0 0 .5rem;
        }

        .notification-card a {
          color: var(--brand-500);
          font-weight: 600;
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
        }

        .stTabs [data-baseweb="tab"]:hover {
          color: var(--brand-500);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
          color: var(--brand-500);
        }

        .stTabs [data-baseweb="tab-highlight"] {
          background: rgba(58,91,255,0.14);
        }

        @media (max-width: 1024px) {
          .block-container {
            padding: 2rem 1.5rem 2.4rem;
          }

          .hero {
            padding: 32px 32px;
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
            if st.button(f"{icon} {label}", key=f"quick_nav_{page_key}"):
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
  <h1>{html.escape(APP_TITLE)}</h1>
  <p>{html.escape(APP_SUBTITLE)}</p>
  <p style='margin-top:10px; font-size:15px; color:var(--text-muted);'>{html.escape(APP_TAGLINE)}</p>
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
        query = st.text_input("请输入问题", value="我想报销, 核心期刊的标准是什么？")
        topk = st.slider("返回引用数量", 1, 5, 3)
        if st.button("检索并生成回答", type="primary"):
            with st.spinner("正在检索权威资料并汇总要点…"):
                hits = rag_search(query, DOCS, VEC, MAT, topk=topk)
                answer = synthesize_answer(query, hits)
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            st.markdown("**引用来源**")
            for h in hits:
                st.markdown(
                    dedent(
                        f"""
<div class='source-card'>
  <b>{html.escape(h['title'])}</b>
  <div class='source-meta'>部门: {html.escape(h['dept'])} ｜ 相似度: {h['score']:.2f}</div>
  <div style='margin-top: 8px'>{html.escape(h['chunk'])}</div>
  <div style='margin-top: 10px'><a href='{h['url']}' target='_blank'>打开原文</a></div>
</div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

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
            threshold = st.slider("推送阈值(相似度)", 0.0, 1.0, 0.18, 0.01)

    if st.button("生成智能推送", type="primary"):
        texts = [f"{n['title']} {n['content']} {' '.join(n['tags'])}" for n in NOTICES]
        order, sims = similarity_match(texts, interests + [identity, school])
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.markdown("### 推荐卡片")
        delivered = 0
        for idx in order:
            if sims[idx] >= threshold:
                notice = NOTICES[idx]
                delay = 0.12 * delivered + 0.15
                st.markdown(
                    dedent(
                        f"""
<div class='notification-card fade-in' style='animation-delay: {delay:.2f}s;'>
  <h4>{html.escape(notice['title'])}</h4>
  <div class='notification-meta'>
    来自: {html.escape(notice['dept'])} ｜ 发布: {notice['publish_time']} ｜ 标签: {', '.join(notice['tags'])}
  </div>
  <div style='margin-top: 10px'>{html.escape(notice['content'])}</div>
  <div style='margin-top: 10px'>截止: {pretty_deadline(notice['deadline'])}</div>
  <div style='margin-top: 12px'><a href='{notice['url']}' target='_blank'>查看详情</a></div>
</div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
                delivered += 1
        if delivered == 0:
            st.markdown(
                dedent(
                    """
<div class='callout-card fade-in' style='animation-delay: 0.15s;'>
  <h4>暂无符合阈值的推送</h4>
  <p>尝试降低相似度阈值, 或选择更多兴趣关键词。</p>
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
        fig_hot.update_layout(
            template="plotly_white",
            xaxis_title="问题",
            yaxis_title="次数",
            xaxis_tickangle=-30,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12),
            margin=dict(l=30, r=10, t=60, b=60),
        )
        st.plotly_chart(fig_hot, use_container_width=True)

    with right:
        if not top_fail.empty:
            fig_fail = px.bar(top_fail, x="q", y="fails", title="Top5 查询失败", text_auto=True)
            fig_fail.update_layout(
                template="plotly_white",
                xaxis_title="问题",
                yaxis_title="失败次数",
                xaxis_tickangle=-30,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12),
                margin=dict(l=30, r=10, t=60, b=60),
            )
            st.plotly_chart(fig_fail, use_container_width=True)
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
