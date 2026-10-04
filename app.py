# -*- coding: utf-8 -*-
"""
输血是否有效预测系统（Streamlit Web 应用）
================================================================================
基于训练好的最优模型（AdaBoost）构建的在线预测应用，可直接部署到
Streamlit Community Cloud。

运行方式：
    本地:   streamlit run app.py
    云端:   将本文件夹上传至 GitHub 仓库 → Streamlit Community Cloud 关联部署
================================================================================
"""
import io
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import os
import streamlit.components.v1 as components

# 截图依赖的 html2canvas 版本（离线打包于 capture_component/html2canvas.min.js）。
# 该库为外部第三方依赖，版本固定，便于复现与排障；版本更新时同步本常量与打包文件。
HTML2CANVAS_VERSION = "1.4.1"

# ----------------------------------------------------------------------------
# 0b. 屏幕截图（位图 PNG，抓取屏幕上真实渲染的画面）
# ----------------------------------------------------------------------------
# 与「矢量 SVG 概率图」不同：本功能直接对浏览器中**实际渲染的 DOM** 做位图截图
# （html2canvas，离线打包版本见 HTML2CANVAS_VERSION），导出与界面所见逐像素一致的 PNG。
# html2canvas 随部署包本地落地，运行时不依赖任何外网 CDN；截图与下载全部在页面内的同源
# iframe 中完成，无需回传 Python。
_CAP_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "capture_component", "html2canvas.min.js")

# 截图 iframe 文档模板：__LIB__ 处内联 html2canvas 源码。关键点：
#   1) iframe 与主页面同源，可直接读取 window.parent.document 取得真实渲染画面；
#   2) 目标选「主内容块」stMainBlockContainer（其高度=内容完整高度），从而一次性
#      截取**整页结果内容**（全部小节），不因当前滚动位置而截断；
#   3) ignoreElements 排除「截图按钮自身的 iframe」，避免导出图里出现空白占位。
_CAP_TARGETS = ('[data-testid="stMainBlockContainer"]', '.block-container',
                '[data-testid="stMain"]', '[data-testid="stAppViewContainer"]')
_CAP_HTML_TEMPLATE = r"""
<div style="font:14px -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;padding:2px 0;">
  <button id="wb_cap_btn" type="button" style="color:#374151;background:#ffffff;border:1px solid #d1d5db;border-radius:6px;padding:7px 16px;cursor:pointer;line-height:1.2;font:inherit;">📸 截图当前界面（位图 PNG）</button>
  <span id="wb_cap_msg" style="margin-left:12px;color:#6b7280;"></span>
</div>
<script>__LIB__</script>
<script>
(function(){
  var SEL=__TARGETS__;
  var btn=document.getElementById('wb_cap_btn');
  var msg=document.getElementById('wb_cap_msg');
  btn.addEventListener('click',function(){
    var doc=window.parent.document;
    var el=null;
    for(var i=0;i<SEL.length;i++){ el=doc.querySelector(SEL[i]); if(el){break;} }
    if(!el){ el=doc.body; }
    var old=btn.textContent; btn.disabled=true; btn.textContent='截图生成中…'; msg.textContent='';
    window.html2canvas(el,{
      scale: 2,
      backgroundColor:'#ffffff',
      useCORS:true, logging:false,
      windowWidth: window.parent.innerWidth,
      windowHeight: window.parent.innerHeight,
      scrollX: 0, scrollY: -window.parent.scrollY,
      ignoreElements:function(e){
        return !!(e.getAttribute && e.getAttribute('data-testid')==='stIFrame');
      }
    }).then(function(c){
      var url=c.toDataURL('image/png');
      var a=document.createElement('a');
      var ts=new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
      a.href=url; a.download='界面截图_'+ts+'.png';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      msg.textContent='✅ 已生成位图（'+c.width+'×'+c.height+'），正在下载';
    }).catch(function(e){ msg.textContent='❌ 截图失败：'+e; })
    .finally(function(){ btn.disabled=false; btn.textContent=old; });
  });
})();
</script>
"""


@st.cache_data(show_spinner=False)
def _load_html2canvas() -> str:
    """读取本地 html2canvas 源码（缓存，避免每次重跑重复读盘）。"""
    try:
        with open(_CAP_LIB_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_capture_html(lib: str) -> str:
    """把内联的 html2canvas 源码与目标选择器注入截图 iframe 文档模板。

    好处：iframe 与主页面同源（sandbox 含 allow-same-origin），可直接读取
    window.parent.document；固定高度即刻可见，规避自定义组件的就绪/高度握手；
    且 sandbox 含 allow-downloads，截图完成后可在 iframe 内直接触发 PNG 下载。
    """
    import json as _json
    return (_CAP_HTML_TEMPLATE
            .replace("__LIB__", lib)
            .replace("__TARGETS__", _json.dumps(list(_CAP_TARGETS))))


def screen_capture_button(key=None):
    """在页面底部放置「截图」按钮：把屏幕上真实渲染的内容截成位图 PNG 并下载。

    采用 st.components.v1.html（固定高度、同源 iframe）承载按钮与截图逻辑；html2canvas
    在 iframe 内完成截图后直接触发下载。key 参数保留以兼容各页面的调用签名（当前未使用）。
    """
    lib = _load_html2canvas()
    if not lib:
        st.caption("（截图功能不可用：未找到本地 html2canvas 库）")
        return
    components.html(_build_capture_html(lib), height=72, scrolling=False)


# ----------------------------------------------------------------------------
# 0. 页面基础配置（必须在任何 st 命令之前）
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="输血是否有效预测系统",
    page_icon="🩸",
    layout="wide",
    # auto：桌面端展开侧边栏，移动端自动收起，避免首屏被导航遮住正文
    initial_sidebar_state="auto",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PKL = BASE_DIR / "model" / "predict.pkl"

# ----------------------------------------------------------------------------
# 0b. 视觉主题：全应用唯一定义处（配色 / 排版 / 组件样式 / 响应式）
# ----------------------------------------------------------------------------
# 配色与 .streamlit/config.toml 的 primaryColor 保持一致，避免「两套红」。
BRAND = "#b02a1f"        # 主色（输血主题深红）
BRAND_DARK = "#8c1f16"
OK_COLOR = "#15803d"     # 阴性 / 正常
ALERT_COLOR = "#c0392b"  # 阳性 / 需关注
MUTED = "#64748b"

# SHAP 背景抽样种子（由训练脚本的 RandomSeeds.SHAP 注入，保证两端一致）
SHAP_SEED = 0

# 全局 CSS：所有页面共享一份，避免逐页重复定义样式
THEME_CSS = """
<style>
/* ===== 设计令牌（极简 / 中性；品牌红仅作唯一强调色） ===== */
:root{
  --brand:#b02a1f; --brand-dark:#8c1f16; --brand-soft:rgba(176,42,31,.06);
  --ok:#15803d; --alert:#c0392b;
  --line:#eceef1; --ink:#1f2937; --muted:#6b7280;
  --surface:#ffffff; --bg:#fafafa; --radius:10px;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",
    "PingFang SC","Noto Sans CJK SC","Hiragino Sans GB",Roboto,Helvetica,Arial,sans-serif;
}
[data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stMainBlockContainer"]{
  padding-top:2rem; padding-bottom:3.5rem; max-width:1140px;
}
/* ---------- 标题排版 ---------- */
h1,h2,h3,h4{ color:var(--ink); font-weight:650; letter-spacing:0; }
h2{ font-size:1.35rem; }
h4{ font-size:1rem; margin:.95rem 0 .25rem; }
[data-testid="stCaptionContainer"] p{ color:var(--muted); font-size:.82rem; }
/* ---------- 统一页头（扁平、留白） ---------- */
.wb-hero{
  display:flex; align-items:center; gap:.9rem; padding:.95rem 1.15rem; margin-bottom:1.4rem;
  background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--brand);
  border-radius:var(--radius);
}
.wb-hero-ico{ font-size:1.7rem; line-height:1; }
.wb-hero h1{ font-size:1.4rem; margin:0; font-weight:700; }
.wb-hero p{ margin:.25rem 0 0; color:var(--muted); font-size:.86rem; }
.wb-chips{ margin-top:.55rem; display:flex; flex-wrap:wrap; gap:.4rem; }
.wb-chip{
  display:inline-block; padding:.16rem .5rem; border-radius:6px; font-size:.74rem;
  color:var(--muted); background:#f3f4f6; border:1px solid var(--line);
}
/* ---------- 卡片 / 表单 ---------- */
[data-testid="stForm"], [data-testid="stExpander"] details{
  background:var(--surface); border:1px solid var(--line) !important;
  border-radius:var(--radius) !important;
}
[data-testid="stForm"]{ padding:.9rem 1.1rem .6rem; }
[data-testid="stExpander"] details{ box-shadow:none !important; }
/* 单样本表单：压缩输入间距，让分类/连续指标更紧凑（移动端由媒体查询改为竖排） */
[data-testid="stForm"] .stVerticalBlock{ gap:.5rem !important; }
[data-testid="stForm"] .stWidget{ margin-bottom:.25rem !important; }
[data-testid="stForm"] h4{ margin:.4rem 0 .1rem; }
[data-testid="stForm"] [data-testid="stCaptionContainer"] p{ margin-bottom:.15rem; }
/* 指标卡：自绘 HTML（.wb-metric），不依赖 Streamlit 容器实现细节 */
.wb-metric{
  border:1px solid var(--line); border-radius:var(--radius); background:var(--surface);
  padding:.9rem 1rem .95rem;
}
.wb-metric-lbl{ color:var(--muted); font-size:.78rem; font-weight:600; line-height:1.4; }
.wb-metric-val{ color:var(--ink); font-size:1.5rem; font-weight:700; line-height:1.4; }
.wb-metric-hint{ color:var(--muted); font-size:.72rem; margin-top:.2rem; }
/* ---------- 按钮（扁平、无渐变、无重阴影） ----------
   1.54 实测 DOM：按钮本体带 data-testid="stBaseButton-*"，用 kind 属性区分主/次；
   外层容器为 stButton / stDownloadButton / stFormSubmitButton（按钮并非其直接子元素）。 */
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button{
  border-radius:8px; font-weight:600; transition:background .15s ease, border-color .15s ease;
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover{ filter:brightness(.97); }
button[kind="primary"], button[kind="primaryFormSubmit"]{
  border:none !important; background:var(--brand) !important; color:#fff !important;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover{
  background:var(--brand-dark) !important;
}
/* ---------- 表单控件 ---------- */
div[data-baseweb="input"] input, div[data-baseweb="select"] > div,
div[data-baseweb="base-input"]{ border-radius:8px !important; }
/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child{ padding-top:1.1rem; }
.wb-brand{ display:flex; align-items:center; gap:.55rem; padding:.2rem 0 .6rem; }
.wb-brand .ico{ font-size:1.4rem; line-height:1; }
.wb-brand .nm{ font-size:1rem; font-weight:700; color:var(--ink); line-height:1.2; }
.wb-brand .sub{ font-size:.74rem; color:var(--muted); }
.wb-card{
  border:1px solid var(--line); border-radius:var(--radius); background:var(--surface);
  padding:.7rem .85rem; margin:.3rem 0 .7rem;
}
.wb-card .row{
  display:flex; justify-content:space-between; gap:.6rem; font-size:.8rem; padding:.18rem 0;
  border-bottom:1px solid #f3f4f6;
}
.wb-card .row:last-child{ border-bottom:none; }
.wb-card .row span:first-child{ color:var(--muted); }
.wb-card .row span:last-child{ color:var(--ink); font-weight:600; text-align:right; }
/* 侧边栏导航：单选卡片化（:has 不支持时自动退化为默认外观） */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:.22rem; }
[data-testid="stSidebar"] [role="radiogroup"] > label{
  display:flex; align-items:center; padding:.34rem .7rem; border-radius:8px;
  border:1px solid transparent; font-weight:600; font-size:.88rem;
  color:var(--ink); cursor:pointer; transition:background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover{ background:#f5f5f6; }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked){
  background:var(--brand-soft); border-color:rgba(176,42,31,.30); color:var(--brand-dark);
}
/* ---------- 结论提示条 / 概率条 ---------- */
.wb-note{
  border:1px solid var(--line); border-left:3px solid var(--brand); background:var(--surface);
  border-radius:8px; padding:.65rem .9rem; font-size:.86rem; color:var(--ink); line-height:1.65;
}
.wb-note.ok{ border-left-color:var(--ok); }
.wb-prob{ display:block; width:100%; max-width:820px; height:auto; margin:0 auto; }
/* ---------- 响应式（移动端 / 窄屏） ---------- */
@media (max-width:768px){
  [data-testid="stMainBlockContainer"]{
    padding-left:.9rem; padding-right:.9rem; padding-top:1.2rem;
  }
  .wb-hero{ flex-direction:column; align-items:flex-start; gap:.5rem; padding:.9rem 1rem; }
  .wb-hero h1{ font-size:1.2rem; }
  .wb-hero-ico{ font-size:1.5rem; }
  .wb-hero p{ font-size:.82rem; }
  .wb-metric-val{ font-size:1.3rem; }
  [data-testid="stSidebar"] [role="radiogroup"] > label{ font-size:.84rem; padding:.3rem .6rem; }
  .wb-metric{ padding:.75rem .85rem .8rem; }
  /* 多列容器在窄屏改为竖排，避免手机端输入框/卡片被挤压换行 */
  [data-testid="stColumns"]{ flex-direction:column !important; }
  [data-testid="stColumns"] > div{
    width:100% !important; max-width:100% !important; min-width:0 !important;
    flex:1 1 100% !important;
  }
}
</style>
"""


def apply_theme():
    """注入全局主题样式（每次 rerun 调一次即可）。"""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def _full_width():
    """返回「撑满容器宽度」的兼容参数：新版用 width，旧版回退 use_container_width。"""
    try:
        import inspect as _inspect
        return ({"width": "stretch"}
                if "width" in _inspect.signature(st.dataframe).parameters
                else {"use_container_width": True})
    except Exception:
        return {"use_container_width": True}


FW = _full_width()


def page_header(icon, title, subtitle="", badges=()):
    """统一页头：图标 + 标题 + 说明 + 徽章，全应用风格一致。"""
    parts = [f'<div class="wb-hero"><div class="wb-hero-ico">{icon}</div><div>',
             f"<h1>{title}</h1>"]
    if subtitle:
        parts.append(f"<p>{subtitle}</p>")
    if badges:
        chips = "".join(f'<span class="wb-chip">{b}</span>' for b in badges)
        parts.append(f'<div class="wb-chips">{chips}</div>')
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(icon, title, desc=""):
    """统一小节标题（配 caption 说明，替代散落的 markdown 标题）。"""
    st.markdown(f"#### {icon} {title}")
    if desc:
        st.caption(desc)


def metric_card(label, value, help=None):
    """统一指标卡：自绘 HTML，圆角/配色/字号全站一致，且不依赖 Streamlit 容器内部实现。"""
    hint = f'<div class="wb-metric-hint">{help}</div>' if help else ""
    st.markdown(
        '<div class="wb-metric">'
        f'<div class="wb-metric-lbl">{label}</div>'
        f'<div class="wb-metric-val">{value}</div>'
        f"{hint}</div>",
        unsafe_allow_html=True)


def show_fig(fig):
    """统一图形渲染：撑满容器 + 渲染后释放图形，避免多次 rerun 累积占用内存。"""
    if fig is None:
        return
    st.pyplot(fig, clear_figure=True, **FW)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 1. 模型加载（Streamlit 缓存，仅加载一次）
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="正在加载预测模型……")
def load_model_pkg():
    if not MODEL_PKL.exists():
        st.error(f"模型文件不存在：{MODEL_PKL}，请确认已运行 pack_model.py 打包。")
        st.stop()
    with open(MODEL_PKL, "rb") as f:
        pkg = pickle.load(f)
    return pkg


PKG = load_model_pkg()
META = PKG["meta"]
CALIBRATOR = META.get("calibrator")            # 可选重校准器（Platt LR / IsotonicRegression 拟合对象）
CALIBRATION_METHOD = META.get("calibration_method")  # 'platt' | 'isotonic' | None
MODEL = PKG["model"]
SCALER = PKG["scaler"]
THRESHOLD = META["threshold"]
FINAL_FEATURES = META["final_features"]
CAT_BINARY = META["cat_binary_cols"]
CAT_MULTI = META["cat_multi_cols"]
TRAIN_MEANS = PKG["train_means"]
# 背景样本（训练/测试集抽样）：bg_raw=原始量纲（用于展示与着色），bg_scaled=标准化后（喂模型）
BG_RAW = PKG.get("bg_raw")
BG_SCALED = PKG.get("bg_scaled")
CLASS_NAMES = META["class_names"]

# 输入字段定义（来自模型包 meta，自动适配本次训练结果）
CONT_FIELDS = META.get("cont_fields", [])   # [{col, unit, default, min, max, step, hint,
                                            #   train_min, train_max, bound_lower, bound_upper}]
CAT_FIELDS = META.get("cat_fields", [])     # [{col, options, hint}]

# 连续指标「合理取值边界」系数：边界 = 训练集最小/最大值的该倍数（默认 1.1）。
# 取值超出边界即视为数据可能不准确（录入错误或异常），预测页会给出提示。
# 系数来自 config.Config.STREAMLIT_INPUT_BOUND_COEFF，由 generate_streamlit_deploy
# 在打包时注入（下方 1.0 占位符替换为具体数值）。
STREAMLIT_INPUT_BOUND_COEFF = 1.0
for _cf in CONT_FIELDS:
    _tmin = _cf.get("train_min")
    _tmax = _cf.get("train_max")
    if _tmin is None or _tmax is None:
        # 模型包未记录真实训练极值时的兜底：用其自带 min/max（通常为 1%/99% 分位）推算
        _tmin = float(_cf.get("min", 0.0))
        _tmax = float(_cf.get("max", 1.0))
    _cf["train_min"] = _tmin
    _cf["train_max"] = _tmax
    # 边界为「包络训练极值的 ±(系数−1) 容差」：下界≈最小值÷系数，上界=最大值×系数
    _cf["bound_lower"] = _tmin / STREAMLIT_INPUT_BOUND_COEFF
    _cf["bound_upper"] = _tmax * STREAMLIT_INPUT_BOUND_COEFF
    # 默认中位数收敛进边界（防御性；边界仅作展示，不限制输入也不作提示）
    _def = float(_cf.get("default", (_tmin + _tmax) / 2))
    _cf["default"] = min(max(_def, _cf["bound_lower"]), _cf["bound_upper"])
    # 始终刷新提示文案（覆盖模型包中可能残留的旧版取值说明）；边界仅作展示与越界提示，不再限制输入
    _cf["hint"] = f"取值范围（{_cf['bound_lower']:.2f}, {_cf['bound_upper']:.2f}）。"
ALL_RAW_COLS = [f["col"] for f in CONT_FIELDS] + [f["col"] for f in CAT_FIELDS]
TEMPLATE_COLS = ALL_RAW_COLS + [META["target_col"]]

# ----------------------------------------------------------------------------
# 2.1 特征工程（与训练脚本完全一致的编码逻辑）
# ----------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """原始临床输入 → 独热编码后的完整特征矩阵（含全部候选列）。

    与训练脚本一致：
    - 二分类变量：保留 0/1 单列
    - 多分类变量：drop-first 独热编码（类别 0 为参考类）
    """
    out = df[CAT_BINARY].round().astype(int).copy()
    for c in CAT_MULTI:
        for v in [1, 2, 3]:  # 训练集中各多分类列的最大类别
            out[f"{c}_{float(v)}"] = (df[c] == v).astype(int)
    rest = [c for c in df.columns if c not in CAT_BINARY + CAT_MULTI]
    return pd.concat([out, df[rest]], axis=1)


def prepare_X(df_raw: pd.DataFrame) -> pd.DataFrame:
    """编码 → 选取最终特征 → 缺失值均值填充 → 返回顺序正确的特征矩阵。"""
    X_enc = encode_features(df_raw)
    missing = [c for c in FINAL_FEATURES if c not in X_enc.columns]
    if missing:
        raise ValueError(f"缺少必要输入列: {missing}")
    X = X_enc[FINAL_FEATURES].astype(float)
    for c in FINAL_FEATURES:
        if X[c].isna().any():
            X[c] = X[c].fillna(TRAIN_MEANS[c])
    return X


def prepare_X_scaled(df_raw: pd.DataFrame) -> pd.DataFrame:
    """prepare_X 之后再做 Z-score 标准化，返回模型真正吃到的矩阵（SHAP 计算口径）。"""
    X = prepare_X(df_raw)
    return pd.DataFrame(SCALER.transform(X), columns=FINAL_FEATURES, index=X.index)


def predict(df_raw: pd.DataFrame) -> np.ndarray:
    """输入原始特征 DataFrame → 返回『阳性』概率数组（已应用可选重校准映射）。"""
    Xs = prepare_X_scaled(df_raw)
    p = MODEL.predict_proba(Xs)[:, 1]
    return _apply_calibration(p)


def _apply_calibration(p):
    """若部署包含重校准器，将原始概率映射为校准概率；否则原样返回（无泄漏、纯训练集拟合）。"""
    if CALIBRATOR is None or CALIBRATION_METHOD is None:
        return np.asarray(p, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    if str(CALIBRATION_METHOD).lower() == "platt":
        lp = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.asarray(CALIBRATOR.predict_proba(lp)[:, 1], dtype=float)
    return np.asarray(CALIBRATOR.predict(p), dtype=float)


# ----------------------------------------------------------------------------
# 3. 结果可视化辅助
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _detect_cjk_font():
    """探测可用中文字体（结果缓存，避免每次 rerun 重复扫描系统字体库）。"""
    from matplotlib import font_manager
    for f in ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
              "WenQuanYi Zen Hei", "AR PL UMing CN", "Droid Sans Fallback"]:
        try:
            font_manager.findfont(f, fallback_to_default=False)
            return f
        except Exception:
            continue
    return None


def _get_plt():
    """返回已配置中文字体的 pyplot（Agg 后端，服务端渲染）。

    Returns:
        (plt, font_ok) —— font_ok 为 None 表示运行环境没有中文字体，
        图中中文会显示为方框，由调用方给出提示。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_ok = _detect_cjk_font()
    if font_ok:
        plt.rcParams["font.sans-serif"] = [font_ok]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, font_ok
def make_linear_prob(prob: float):
    """返回 SVG 直线型概率条（矢量、随容器自适应，窄屏同样清晰）。

    相比原半环仪表盘：改成水平进度条，决策阈值刻度线更直观，且更省竖向空间，
    在「本例 SHAP 归因」之上占据一条窄带即可。
    """
    v = max(0.0, min(1.0, float(prob)))
    color = ALERT_COLOR if v >= THRESHOLD else OK_COLOR
    pos_label = CLASS_NAMES.get(1, "阳性")
    W, H = 1000, 150
    x0, x1, y, hh = 24.0, 976.0, 62.0, 24.0
    track_w = x1 - x0
    fill_w = track_w * v
    t = float(THRESHOLD)
    tx = x0 + track_w * t if 0.0 < t < 1.0 else None
    parts = [f'<svg class="wb-prob" viewBox="0 0 {W} {H}" role="img"'
             f' aria-label="{pos_label}概率 {v * 100:.1f}%">']
    # 标题（顶部居中，不与阈值标记重叠）
    parts.append(f'<text x="{x0 + track_w/2:.1f}" y="36" text-anchor="middle" '
                 f'font-size="30" font-weight="800" fill="{color}">{pos_label}概率 {v * 100:.1f}%</text>')
    # 进度条轨道 + 填充
    parts.append(f'<rect x="{x0}" y="{y}" width="{track_w}" height="{hh}" '
                 f'rx="{hh/2:.0f}" fill="#eef1f4"/>')
    parts.append(f'<rect x="{x0}" y="{y}" width="{fill_w:.1f}" height="{hh}" '
                 f'rx="{hh/2:.0f}" fill="{color}"/>')
    # 阈值刻度线（贯穿条身）+ 标签（放在条下方，避免与标题重叠）
    if tx is not None:
        parts.append(f'<line x1="{tx:.1f}" y1="{y-10:.1f}" x2="{tx:.1f}" y2="{y+hh+10:.1f}" '
                     f'stroke="#475569" stroke-width="2"/>')
        parts.append(f'<text x="{tx:.1f}" y="{y+hh+28:.1f}" text-anchor="middle" '
                     f'font-size="18" fill="#475569">阈值 {t*100:.1f}%</text>')
    # 0% / 100%（最底部）
    parts.append(f'<text x="{x0}" y="{y+hh+46:.1f}" font-size="20" fill="#64748b">0%</text>')
    parts.append(f'<text x="{x1}" y="{y+hh+46:.1f}" text-anchor="end" '
                 f'font-size="20" fill="#64748b">100%</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# 3b. SHAP 可解释性（全局解释 / 单样本归因 / 特征依赖）
# ----------------------------------------------------------------------------
def shap_available() -> bool:
    """运行环境是否装了 shap（缺失时页面降级提示，不影响预测功能）。"""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="正在构建 SHAP 解释器……")
def get_shap_explainer():
    """树模型优先用 TreeExplainer（快且精确），否则回退 KernelExplainer。"""
    import shap

    name = type(MODEL).__name__
    tree_like = ("CatBoost" in name or "XGB" in name or "LGBM" in name
                 or name in ("RandomForestClassifier", "GradientBoostingClassifier",
                             "DecisionTreeClassifier", "ExtraTreesClassifier")
                 or hasattr(MODEL, "get_booster"))
    if tree_like:
        try:
            return shap.TreeExplainer(MODEL)
        except Exception:
            pass
    bg = BG_SCALED if BG_SCALED is not None else None
    return shap.KernelExplainer(
        lambda X: MODEL.predict_proba(pd.DataFrame(np.asarray(X), columns=FINAL_FEATURES))[:, 1],
        shap.sample(bg, min(50, len(bg)), random_state=SHAP_SEED))


def compute_shap(X_scaled: pd.DataFrame):
    """计算 SHAP 值。

    Returns:
        (sv, base_value)：sv 形状 (n_samples, n_features)，取正类口径；
        base_value 为基线值（标量）。
    """
    explainer = get_shap_explainer()
    try:
        exp = explainer(X_scaled)
        sv = np.asarray(exp.values)
        bv = getattr(exp, "base_values", None)
        bv = np.asarray(bv) if bv is not None else None
    except Exception:
        out = explainer.shap_values(X_scaled)
        sv = np.asarray(out[1] if isinstance(out, list) else out)
        bv = None
    if sv.ndim == 3:                       # (n, f, n_classes) → 取正类
        sv = sv[:, :, 1]
    if bv is None:
        bv = 0.0
    elif bv.ndim == 0:
        bv = float(bv)
    elif bv.ndim == 1:
        bv = float(bv[1]) if bv.size > 1 else float(bv[0])
    else:                                  # (n_samples, n_classes)
        bv = float(bv[0, 1]) if bv.shape[1] > 1 else float(bv[0, 0])
    return sv, float(bv)


@st.cache_data(show_spinner="正在计算全局 SHAP 值（首次较慢）……")
def global_shap():
    """背景样本上的全局 SHAP 结果：(sv, base_value, 平均绝对SHAP Series)。"""
    sv, bv = compute_shap(BG_SCALED)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=FINAL_FEATURES).sort_values(
        ascending=False)
    return sv, float(bv), mean_abs


def shap_summary_fig(sv, bg_df, max_display, plot_type=None):
    """SHAP 摘要图：plot_type=None 为蜂群点图，'bar' 为平均绝对 SHAP 条形图。"""
    import shap
    plt, _ = _get_plt()
    plt.figure()
    shap.summary_plot(sv, bg_df, plot_type=plot_type, show=False,
                      max_display=max_display)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def shap_dependence_fig(sv, bg_df, feat):
    """单特征依赖图：特征值 vs 其 SHAP 值（自动选交互特征着色）。"""
    import shap
    plt, _ = _get_plt()
    plt.figure()
    shap.dependence_plot(feat, sv, bg_df, show=False, interaction_index="auto")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def shap_waterfall_fig(sv_row, base_value, raw_row, max_display=10):
    """单样本瀑布图：从基线值出发，逐特征累加贡献，得到该样本的预测值。

    raw_row 用原始量纲（如 HGB=62 g/L）展示，便于临床解读；SHAP 值本身由标准化
    后的矩阵算得——树模型下两者等价（标准化是单调线性变换，分裂完全一致）。

    ⚠️ shap.waterfall_plot 的 max_display 必须为 int（不能为 None），否则
    min(None, len) 会抛 TypeError。这里默认 10，调用方也都传 int。
    """
    import shap
    plt, _ = _get_plt()
    n = len(FINAL_FEATURES)
    _max = int(max_display) if max_display else min(10, n)
    exp = shap.Explanation(values=np.asarray(sv_row, dtype=float),
                           base_values=float(base_value),
                           data=np.asarray(raw_row, dtype=float),
                           feature_names=list(FINAL_FEATURES))
    plt.figure(figsize=(8, max(4.0, 0.42 * min(n, _max) + 3.0)), dpi=150)
    # SHAP 瀑布图用 U+2212（真减号）绘制数值标签，而 SimHei 等中文字体缺该字形，
    # 会触发 Glyph 警告（且可能显示为方框）。① 调用时抑制该无害警告；
    # ② 绘制后把所有 U+2212 替换为 ASCII 减号（U+002D，中文字体均有），保证渲染清晰。
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        shap.waterfall_plot(exp, show=False, max_display=_max)
    fig = plt.gcf()
    for t in fig.findobj(match=plt.Text):
        s = t.get_text()
        if "\u2212" in s:
            t.set_text(s.replace("\u2212", "-"))
    fig.tight_layout()
    return fig


def shap_contrib_table(sv_row, raw_row):
    """单样本各特征贡献表（按 |SHAP| 降序）。"""
    pos_label, neg_label = CLASS_NAMES.get(1, "阳性"), CLASS_NAMES.get(0, "阴性")
    df = pd.DataFrame({
        "特征": FINAL_FEATURES,
        "该样本取值": [round(float(v), 4) for v in np.asarray(raw_row, dtype=float)],
        "SHAP值": np.asarray(sv_row, dtype=float).round(4),
    })
    df["|SHAP|"] = df["SHAP值"].abs()
    df["影响方向"] = np.where(df["SHAP值"] >= 0, f"↑ 推向{pos_label}", f"↓ 推向{neg_label}")
    return df.sort_values("|SHAP|", ascending=False).drop(columns=["|SHAP|"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. 页面一：单样本预测
# ----------------------------------------------------------------------------
def page_single():
    pos_label = CLASS_NAMES.get(1, "阳性")
    page_header(
        "🩸", "单样本在线预测",
        f"输入该病例的临床指标，模型将给出{pos_label}概率与预测结论。")

    # 重置采用「递增版本号」而非删除键：键变了，控件自然回到默认值，也避免了
    # 「控件实例化后再改其 session_state」的 Streamlit 限制。
    _rn = int(st.session_state.get("single_reset_n", 0))
    # 提示延迟到重跑后弹出，否则会被 st.rerun() 一并清掉
    if st.session_state.get("_single_reset_done"):
        st.session_state["_single_reset_done"] = False
        st.toast("已恢复默认值", icon="🔄")

    # 左右两栏：左侧输入表单（分类/连续指标），右侧图形结果
    _left, _right = st.columns([1, 1])

    with _left:
        # 表单：改动输入不触发重算，仅在点击「开始预测」时推理一次（更快、更省资源）。
        # 桌面端分类/连续指标各用多列网格排布，间距由 THEME_CSS 压缩；移动端自动竖排。
        cat_vals, cont_vals = {}, {}
        with st.form("single_form"):
            if CAT_FIELDS:
                section("🧬", "分类指标", "选择该病例的分类特征取值。")
                _cat_cols = st.columns(min(3, len(CAT_FIELDS)))
                for i, f in enumerate(CAT_FIELDS):
                    with _cat_cols[i % len(_cat_cols)]:
                        cat_vals[f["col"]] = st.selectbox(
                            f["col"], f["options"], key=f"in_cat_{f['col']}_{_rn}",
                            help=f.get("hint", ""))

            if CONT_FIELDS:
                section("📈", "连续指标",
                        f"各指标取值范围以（最小值, 最大值）格式展示（保留两位小数），为训练集实际范围。")
                _c1, _c2, _c3 = st.columns(3)
                _third = (len(CONT_FIELDS) + 2) // 3
                for i, f in enumerate(CONT_FIELDS):
                    _col = _c1 if i < _third else (_c2 if i < 2 * _third else _c3)
                    with _col:
                        _unit = f.get("unit") or ""
                        cont_vals[f["col"]] = st.number_input(
                            f"{f['col']}" + (f"（{_unit}）" if _unit else ""),

                            value=float(f["default"]), step=float(f["step"]),
                            key=f"in_cont_{f['col']}_{_rn}", help=f.get("hint", ""))

            submitted = st.form_submit_button("🚀 开始预测", type="primary", **FW)

        # 「恢复默认值」放在表单提交按钮下方，视觉上紧随其后
        if st.button("↺ 恢复默认值", key="single_reset", **FW,
                     help="把全部输入恢复为训练集中位数 / 默认类别"):
            st.session_state["single_reset_n"] = _rn + 1
            st.session_state["_single_reset_done"] = True
            for _k in ("last_raw_df", "last_prob"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

    with _right:
        if not submitted:
            st.caption("👈 在左侧填写指标后点击「开始预测」，概率图与本例归因图将显示在此处。")
            return

        df_in = pd.DataFrame([{**cat_vals, **cont_vals}])

        prob = float(predict(df_in)[0])
        hit = prob >= THRESHOLD
        pred_label = CLASS_NAMES.get(1) if hit else CLASS_NAMES.get(0)

        # 缓存本次输入，供「SHAP 可解释性」页直接复用
        st.session_state["last_raw_df"] = df_in
        st.session_state["last_prob"] = prob

        section("📌", "预测结论")
        _mcols = st.columns(2)
        with _mcols[0]:
            metric_card(f"{pos_label}概率", f"{prob * 100:.2f}%")
        with _mcols[1]:
            metric_card("预测类别", pred_label)
        _mcols2 = st.columns(2)
        with _mcols2[0]:
            metric_card("决策阈值", f"{THRESHOLD * 100:.2f}%")
        with _mcols2[1]:
            metric_card("距阈值（百分点）", f"{(prob - THRESHOLD) * 100:+.2f}")

        _near = abs(prob - THRESHOLD) < 0.05
        st.markdown(
            f'<div class="wb-note{" ok" if hit else ""}">模型判定为 <b>{pred_label}</b>：'
            f'{pos_label}概率 {prob * 100:.2f}% {"≥" if hit else "&lt;"} '
            f'决策阈值 {THRESHOLD * 100:.2f}%。'
            + ("<br>⚠️ 概率与阈值接近，本例结论不确定性较高，请结合临床综合判断。"
               if _near else "")
            + "</div>", unsafe_allow_html=True)

        # 概率图（直线型）+ 本例 SHAP 归因
        st.markdown(make_linear_prob(prob), unsafe_allow_html=True)
        if not shap_available():
            section("🔍", "本例 SHAP 归因", "解释这一个病例为什么被判为该结果（逐特征贡献）。")
            st.info("未安装 shap，跳过单样本归因。执行 `pip install shap` 后刷新即可；"
                    "更多图形见「🔍 SHAP 可解释性」页。")
        else:
            section("🔍", "本例 SHAP 归因", "解释这一个病例为什么被判为该结果（逐特征贡献）。")
            with st.spinner("正在计算本例 SHAP 归因……"):
                try:
                    _Xs = prepare_X_scaled(df_in)
                    _sv, _bv = compute_shap(_Xs)
                    _raw = prepare_X(df_in).iloc[0].values
                    _fig = shap_waterfall_fig(_sv[0], _bv, _raw)
                    # 论文级高清导出：先存 300 dpi PNG，再在页面渲染（之后释放图形）
                    _buf = io.BytesIO()
                    _fig.savefig(_buf, dpi=300, bbox_inches="tight", facecolor="white")
                    show_fig(_fig)
                    st.dataframe(shap_contrib_table(_sv[0], _raw),
                                 hide_index=True, **FW)
                    st.download_button(
                        "⬇️ SHAP 归因图（论文用·300 dpi PNG）",
                        data=_buf.getvalue(),
                        file_name="SHAP_单样本归因.png", mime="image/png", **FW)
                except Exception as e:
                    st.warning(f"SHAP 单样本归因失败：{e}")

        st.caption("⚠️ 本工具仅作临床辅助决策参考，不能替代专业医师判断；"
                   "预测基于历史病例训练，请结合实际情况综合评估。")

        screen_capture_button(key="cap_single")


# ----------------------------------------------------------------------------
# 5. 页面二：批量预测
# ----------------------------------------------------------------------------
def page_batch():
    page_header(
        "📋", "批量预测",
        "上传 CSV 或使用内置合成示例，一次完成整批推理并导出结果。",
        badges=[f"必需列 {len(ALL_RAW_COLS)} 个",
                f"可选真实标签列 {META['target_col']}"])

    _sample = PKG.get("sample_input")
    _has_sample = _sample is not None and len(_sample) > 0

    _d1, _d2 = st.columns([1, 2])
    with _d1:
        _buf = io.BytesIO()
        pd.DataFrame([{c: "" for c in TEMPLATE_COLS}]).to_csv(
            _buf, index=False, encoding="utf-8-sig")
        st.download_button("⬇️ 下载输入模板", data=_buf.getvalue(),
                           file_name="sample_input.csv", mime="text/csv", **FW)
    with _d2:
        st.caption("数据来源：可直接用合成示例体验，或上传自己的 CSV（第一行为列名，"
                   "列名需与模板一致）。")

    _opts = (["🧪 内置合成示例"] if _has_sample else []) + ["📤 上传 CSV"]
    src = st.segmented_control("数据来源", _opts, default=_opts[0],
                               key="batch_src", label_visibility="collapsed")

    df = None
    if src is None:
        st.caption("请选择数据来源。")
        return
    if src.startswith("📤"):
        uploaded = st.file_uploader("选择 CSV 文件", type=["csv"], key="batch_csv")
        if uploaded is None:
            st.caption("尚未选择文件。第一行为列名，列名需与模板一致。")
            return
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as e:
            st.error(f"CSV 解析失败：{e}")
            return
    else:
        df = _sample.copy()
        st.caption("已加载**合成内置示例数据**（由训练分布随机生成，不含任何真实患者信息）。")

    if df is None or df.shape[0] == 0:
        st.warning("文件为空，未发现数据行。")
        return

    missing = [c for c in ALL_RAW_COLS if c not in set(df.columns)]
    if missing:
        st.error("缺少必要列：" + "、".join(map(str, missing))
                 + f"（共缺 {len(missing)} 列）。请下载模板后按模板列名填写。")
        return

    df_in = df[ALL_RAW_COLS].copy()
    n = len(df_in)

    # 分批推理 + 进度反馈：大文件不再表现为「卡住无响应」
    _chunk = 500
    with st.status(f"正在批量推理（{n} 条）……", expanded=False) as _status:
        if n > _chunk:
            _bar = st.progress(0.0)
            proba = np.empty(n, dtype=float)
            for _s in range(0, n, _chunk):
                _e = min(_s + _chunk, n)
                proba[_s:_e] = predict(df_in.iloc[_s:_e])
                _bar.progress(_e / n, text=f"已完成 {_e}/{n}")
            _bar.empty()
        else:
            proba = predict(df_in)
        _status.update(label=f"批量推理完成（{n} 条）", state="complete")

    out = df.copy()
    out["预测概率"] = np.round(proba, 6)
    out["预测结果"] = np.where(proba >= THRESHOLD,
                               CLASS_NAMES.get(1), CLASS_NAMES.get(0))

    section("📊", "结果汇总")
    _n_pos = int((proba >= THRESHOLD).sum())
    _mcols = st.columns(4)
    with _mcols[0]:
        metric_card("样本数", f"{n}")
    with _mcols[1]:
        metric_card(f"{CLASS_NAMES.get(1)}数", f"{_n_pos}")
    with _mcols[2]:
        metric_card(f"{CLASS_NAMES.get(1)}占比", f"{_n_pos / n * 100:.2f}%")

    # 若上传数据带真实标签列，顺带给出 AUC 作为一致性校验
    _auc = None
    if META["target_col"] in df.columns:
        try:
            from sklearn.metrics import roc_auc_score
            _auc = float(roc_auc_score(df[META["target_col"]].astype(int), proba))
        except Exception:
            _auc = None
    with _mcols[3]:
        if _auc is None:
            metric_card("真实标签 AUC", "—", help="上传文件包含真实标签列时自动计算")
        else:
            metric_card("真实标签 AUC", f"{_auc:.4f}")

    _show = min(n, 200)
    section("🧾", "结果预览", f"最多展示前 {_show} 行；完整结果请下载 CSV。")
    st.dataframe(out.head(_show), hide_index=True,
                 height=min(420, 42 + _show * 35), **FW)

    _rbuf = io.BytesIO()
    out.to_csv(_rbuf, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ 下载预测结果（CSV）", data=_rbuf.getvalue(),
                       file_name="prediction_result.csv", mime="text/csv",
                       type="primary", **FW)

    screen_capture_button(key="cap_batch")


# ----------------------------------------------------------------------------
# 6. 页面三：SHAP 可解释性分析
# ----------------------------------------------------------------------------
def page_shap():
    pos_label, neg_label = CLASS_NAMES.get(1, "阳性"), CLASS_NAMES.get(0, "阴性")
    page_header(
        "🔍", "SHAP 可解释性分析",
        f"SHAP 值衡量每个特征对单次预测的边际贡献：正值把预测推向{pos_label}，"
        f"负值推向{neg_label}；绝对值越大影响越强。",
        badges=[f"建模特征 {len(FINAL_FEATURES)} 个",
                "背景样本为合成数据"])

    if not shap_available():
        st.error("当前环境未安装 shap，无法进行可解释性分析。")
        st.code("pip install shap", language="bash")
        return
    if BG_SCALED is None or BG_RAW is None or len(BG_SCALED) == 0:
        st.warning("模型包中缺少背景样本（bg_raw / bg_scaled），无法进行全局 SHAP 分析。"
                   "请用最新版训练脚本重新生成部署包。")
        return

    _, font_ok = _get_plt()
    if font_ok is None:
        st.caption("提示：运行环境未检测到中文字体，图中中文可能显示为方框；"
                   "本地 Windows / macOS 一般正常，云端部署可自行挂载字体。")
    st.caption("🔒 本页背景样本为**合成数据**（由训练分布随机生成），不含任何真实患者信息。")

    tab_g, tab_s, tab_d = st.tabs(["🌐 全局解释", "🧑‍⚕️ 单样本归因", "📈 特征依赖"])

    # ---------------- 全局解释 ----------------
    with tab_g:
        max_display = st.slider("展示特征数", 3, len(FINAL_FEATURES),
                                min(10, len(FINAL_FEATURES)), key="shap_md")
        _f_bar = _f_bee = None
        _bv = _mean_abs = None
        with st.spinner("正在计算全局 SHAP 值（首次较慢，之后会缓存）……"):
            try:
                sv, _bv, _mean_abs = global_shap()
                _f_bar = shap_summary_fig(sv, BG_RAW, max_display, plot_type="bar")
                _f_bee = shap_summary_fig(sv, BG_RAW, max_display)
            except Exception as e:
                st.warning(f"全局 SHAP 计算失败：{e}")
        if _f_bar is not None:
            st.caption(f"基线值（全体样本平均输出）= {_bv:.4f}")
            section("📊", "平均绝对 SHAP 值（全局重要性）")
            show_fig(_f_bar)
            section("🐝", "SHAP 蜂群图", "点的颜色 = 该样本该特征的取值高低。")
            show_fig(_f_bee)
            section("📋", "重要性排序表")
            st.dataframe(pd.DataFrame({"特征": _mean_abs.index,
                                       "平均|SHAP|": _mean_abs.values.round(4)}),
                         hide_index=True, **FW)

    # ---------------- 单样本归因 ----------------
    with tab_s:
        use_last = "last_raw_df" in st.session_state
        _sopts = (["当前输入病例"] if use_last else []) + ["内置合成示例"]
        _sel = st.segmented_control("样本来源", _sopts, default=_sopts[0],
                                    key="shap_src", label_visibility="collapsed")
        if _sel is None:
            _sel = _sopts[0]
        with st.spinner("正在计算单样本 SHAP 归因……"):
            try:
                if use_last and _sel == "当前输入病例":
                    df_one = st.session_state["last_raw_df"]
                    Xs_one = prepare_X_scaled(df_one)           # 走完整编码链路
                    raw_row = prepare_X(df_one).iloc[0].values   # 原始量纲（用于展示）
                    tag = "当前输入病例"
                else:
                    idx = st.slider("内置示例样本序号", 0, len(BG_RAW) - 1, 0,
                                    key="shap_idx")
                    # 内置示例样本已是「编码后」矩阵，无需再走一遍编码
                    Xs_one = BG_SCALED.iloc[idx:idx + 1]
                    raw_row = BG_RAW.iloc[idx].values
                    tag = f"内置示例样本 #{idx}"
                prob_one = float(MODEL.predict_proba(Xs_one)[:, 1][0])
                _hit = prob_one >= THRESHOLD
                sv_one, bv_one = compute_shap(Xs_one)
                st.markdown(
                    f'<div class="wb-note{" ok" if not _hit else ""}">'
                    f'{tag}：预测{pos_label}概率 <b>{prob_one * 100:.2f}%</b>'
                    f'（阈值 {THRESHOLD * 100:.2f}% → '
                    f'<b>{pos_label if _hit else neg_label}</b>）</div>',
                    unsafe_allow_html=True)
                st.write("")
                show_fig(shap_waterfall_fig(sv_one[0], bv_one, raw_row))
                st.dataframe(shap_contrib_table(sv_one[0], raw_row),
                             hide_index=True, **FW)
            except Exception as e:
                st.warning(f"单样本 SHAP 计算失败：{e}")

    # ---------------- 特征依赖 ----------------
    with tab_d:
        try:
            sv, _bv, _mean_abs = global_shap()
            feat = st.selectbox("选择特征", list(_mean_abs.index), key="shap_dep_feat")
            st.caption("横轴为该特征的**原始取值**，纵轴为其 SHAP 值；"
                       "点的颜色自动对应交互最强的另一个特征。")
            show_fig(shap_dependence_fig(sv, BG_RAW, feat))
        except Exception as e:
            st.warning(f"特征依赖图绘制失败：{e}")

    screen_capture_button(key="cap_shap")


# ----------------------------------------------------------------------------
# 7. 页面四：模型信息
# ----------------------------------------------------------------------------
def page_info():
    m = META["metrics"]
    n_cand = int(META.get("n_candidates", len(FINAL_FEATURES)) or len(FINAL_FEATURES))
    page_header(
        "📊", "模型信息与使用说明",
        "本次部署所用模型的性能、输入字段与预测流程说明。",
        badges=[f"{META['model_name']}",
                f"训练样本 {META['train_n']} 例",
                f"外部验证集 {META['ext_datasets']} 个"])

    section("📈", "关键性能")
    _mcols = st.columns(4)
    with _mcols[0]:
        metric_card("测试集 AUC", f"{m['test_auc']:.4f}")
    with _mcols[1]:
        metric_card("测试集 F1", f"{m['test_f1']:.4f}")
    with _mcols[2]:
        metric_card("测试集准确率", f"{m['test_acc']:.4f}")
    with _mcols[3]:
        metric_card("外部验证 AUC", f"{m['ext_auc']:.4f}",
                    help=f"基于 {META['ext_datasets']} 个外部验证集计算")

    section("🧾", "模型概况")
    meta_rows = [
        ("模型算法", META["model_name"]),
        ("任务类型", META["task"]),
        ("特征数量", f"{len(FINAL_FEATURES)} 个（由 {n_cand} 个候选特征经特征筛选保留）"),
        ("输入字段", f"{len(CAT_FIELDS)} 个分类 + {len(CONT_FIELDS)} 个连续"
                     "（仅含筛后模型实际需要的原始变量）"),
        ("决策阈值", f"{THRESHOLD:.4f}（约登指数最大化最佳截断点）"),
        ("训练样本", f"{META['train_n']} 例（训练 {META['train_n_train']} / "
                     f"测试 {META['train_n_test']}）"),
    ]
    st.dataframe(pd.DataFrame(meta_rows, columns=["项目", "内容"]),
                 hide_index=True, **FW)

    section("🗂️", "输入字段说明",
            "下列字段为特征筛选后模型实际需要的原始变量，筛选中被淘汰的变量无需填写；"
            "分类变量中二分类保留 0/1 单列，多分类采用 drop-first 独热编码。")
    st.dataframe(
        pd.DataFrame(
            [{"字段": f["col"], "类型": "分类", "取值/说明": f.get("hint", "")}
             for f in CAT_FIELDS]
            + [{"字段": f["col"], "类型": "连续",
                "取值/说明": f"单位 {f.get('unit')}" if f.get("unit") else ""}
               for f in CONT_FIELDS]
        ),
        hide_index=True, **FW,
    )

    with st.expander(f"建模特征清单（编码后，共 {len(FINAL_FEATURES)} 个）"):
        st.markdown("、".join(map(str, FINAL_FEATURES)))

    section("🔄", "预测流程")
    st.markdown(
        f"原始临床输入 → 分类变量独热编码 → 从 {n_cand} 个候选中选取 "
        f"{len(FINAL_FEATURES)} 个最终特征 → 缺失值均值填充 → Z-score 标准化 → "
        f"{META['model_name']} 输出概率 → 阈值判定"
    )

    section("🚀", "本地部署命令")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")


    screen_capture_button(key="cap_info")


# ----------------------------------------------------------------------------
# 8. 侧边栏导航与主入口
# ----------------------------------------------------------------------------
PAGES = ["🩸 单样本预测", "📋 批量预测", "🔍 SHAP 可解释性", "📊 模型信息"]


def _sidebar_model_card():
    """侧边栏模型摘要卡片（HTML 单块渲染，避免多组件堆叠占用纵向空间）。"""
    m = META["metrics"]
    rows = [
        ("模型算法", META["model_name"]),
        ("测试集 AUC", f"{m['test_auc']:.4f}"),
        ("外部验证 AUC", f"{m['ext_auc']:.4f}"),
        ("决策阈值", f"{THRESHOLD:.4f}"),
        ("建模特征", f"{len(FINAL_FEATURES)} 个"),
        ("训练样本", f"{META['train_n']} 例"),
    ]
    body = "".join(f'<div class="row"><span>{k}</span><span>{v}</span></div>'
                   for k, v in rows)
    return f'<div class="wb-card">{body}</div>'


def main():
    apply_theme()
    with st.sidebar:
        st.markdown(
            '<div class="wb-brand"><div class="ico">🩸</div><div>'
            '<div class="nm">输血是否有效预测系统</div>'
            '<div class="sub">机器学习临床预测</div></div></div>',
            unsafe_allow_html=True)
        page = st.radio("功能导航", PAGES, key="nav_page",
                        label_visibility="collapsed")
        st.markdown(_sidebar_model_card(), unsafe_allow_html=True)
        st.caption("⚠️ 仅供临床辅助参考，不替代医师判断")

    if page == PAGES[0]:
        page_single()
    elif page == PAGES[1]:
        page_batch()
    elif page == PAGES[2]:
        page_shap()
    else:
        page_info()


if __name__ == "__main__":
    main()
