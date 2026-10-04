# -*- coding: utf-8 -*-
"""
模型打包脚本 pack_model.py（供重新打包/在新结果目录复用）
================================================================================
将训练好的最优模型 + 预处理器（StandardScaler）+ 特征顺序 + 决策阈值等
全部元数据打包为单一文件 `model/predict.pkl`，供 Streamlit 应用加载。

使用方法（在本地训练环境执行一次即可）：
    python pack_model.py

产物：
    streamlit/model/predict.pkl  —— 单一可部署模型包（自包含，无外部路径依赖）
================================================================================
"""
import os
import pickle

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径配置
# 自动探测结果目录：脚本位于「结果目录/streamlit」子目录时取父目录；否则手动指定
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULT_PARENT = os.path.dirname(_HERE)
OUT_PKL = os.path.join(_HERE, "model", "predict.pkl")

# ---------------------------------------------------------------- 模型元数据
BEST_MODEL_NAME = "CatBoost"
BEST_THRESHOLD = 0.6533
MODEL_PKL = "best_model.pkl"
SCALER_PKL = "scaler.pkl"
# 模型权重可能已被归集到结果根目录下的附属子文件夹（config.MISC_DIRNAME，默认 other），
# 也可能仍在结果根目录；依次探测，取第一个真正存在 best_model.pkl 的位置。
_MISC_SUBDIR = r"other"
RESULT_DIR = r"结果/20261004_115417_分类"
for _d in [_RESULT_PARENT,
           os.path.join(_RESULT_PARENT, _MISC_SUBDIR) if _MISC_SUBDIR else "",
           r"结果/20261004_115417_分类"]:
    if _d and os.path.exists(os.path.join(_d, MODEL_PKL)):
        RESULT_DIR = _d
        break
FINAL_FEATURES = ['血型_2.0', '高血压', 'DBIL', '输血量', 'PLT', 'TBIL', '身高', 'RBC', '性别']
CAT_BINARY_COLS = ['性别', '高血压']
CAT_MULTI_COLS = ['血型']
TARGET_COL = '是否有效'
TRAIN_CSV = r"data\HGB后表-1.csv"   # 训练原始数据（仅用于统计均值/标准差，绝不保留任何原始行）


def _synth_bg(final_features, cat_binary, train_means, train_std, scaler, n_bg, seed):
    """生成**合成** SHAP 背景样本（不含任何真实患者行）。"""
    import numpy as np, pandas as pd
    rng = make_rng(seed)
    raw = {}
    for c in final_features:
        mu = float(train_means.get(c, 0.0))
        sd = float(train_std.get(c, 0.0))
        if sd <= 0:
            sd = max(abs(mu) * 0.1, 1e-3)
        v = rng.normal(mu, sd, n_bg)
        if c in cat_binary:
            v = (v >= 0.5).astype(int)
        else:
            v = np.clip(v, mu - 4 * sd, mu + 4 * sd)
        raw[c] = v
    bg_raw = pd.DataFrame(raw, columns=final_features)
    bg_scaled = pd.DataFrame(scaler.transform(bg_raw), columns=final_features)
    return bg_raw, bg_scaled


def _synth_sample(cont_fields, cat_fields, n, seed):
    """生成**合成**内置示例数据（批量页「加载内置示例」用，不含真实患者）。"""
    import numpy as np, pandas as pd
    rng = make_rng(seed)
    cols = [f["col"] for f in cont_fields] + [f["col"] for f in cat_fields]
    rows = []
    for _ in range(n):
        row = {}
        for cf in cont_fields:
            lo, hi = float(cf.get("min", 0)), float(cf.get("max", 1))
            m = float(cf.get("default", (lo + hi) / 2))
            span = (hi - lo) or 1.0
            row[cf["col"]] = round(float(np.clip(rng.normal(m, span / 6), lo, hi)), 3)
        for f in cat_fields:
            # 从该字段的合法取值域（options）均匀采样，保证生成的分类值落在合法范围
            # （如「血型」含 0/1/2/3，避免 rng.integers(0,2) 只生成 0/1 导致编码失败或空类别）。
            opts = f.get("options")
            if opts:
                opts = list(opts)
                row[f["col"]] = opts[int(rng.integers(0, len(opts)))]
            else:
                row[f["col"]] = int(rng.integers(0, 2))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_package():
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)

    # 1. 加载最优模型与预处理器
    model = joblib.load(os.path.join(RESULT_DIR, MODEL_PKL))
    scaler = joblib.load(os.path.join(RESULT_DIR, SCALER_PKL))
    print(f"[1] 模型类型   : {type(model).__name__}")
    print(f"[2] 预处理器   : {type(scaler).__name__}")

    # 2. 统计训练均值 / 标准差（仅统计用，绝不保留任何原始行）
    train_means, train_std = {}, {}
    sample_input = None
    if os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV, encoding="gbk")
        print(f"[4] 训练数据   : {df.shape[0]} 样本 × {df.shape[1]} 列")
        # 复刻训练集编码：二分类保留单列；多分类 drop-first 独热
        X_enc = df[CAT_BINARY_COLS].round().copy()
        for c in CAT_MULTI_COLS:
            for v in [1, 2, 3]:
                X_enc[f"{c}_{float(v)}"] = (df[c] == v).astype(int)
        rest = [c for c in df.columns if c not in CAT_BINARY_COLS + CAT_MULTI_COLS + [TARGET_COL]]
        X_full = pd.concat([X_enc, df[rest]], axis=1)
        try:
            train_means = X_full[FINAL_FEATURES].mean().to_dict()
            train_std = X_full[FINAL_FEATURES].std().fillna(0.0).to_dict()
        except KeyError as e:
            print("警告：原始数据列与最终特征不完全匹配，跳过统计：", e)
        # 依据训练分布构造合成「内置示例」字段定义
        cont_fields, cat_fields = [], []
        for c in FINAL_FEATURES:
            if c in CAT_BINARY_COLS:
                continue
            s = X_full[c].dropna()
            if s.empty:
                continue
            cont_fields.append({"col": c, "unit": "", "default": float(s.median()),
                                "min": float(s.quantile(0.01)), "max": float(s.quantile(0.99)),
                                "step": 0.1, "hint": ""})
        cat_fields = [{"col": c, "options": sorted(df[c].dropna().unique().tolist())}
                      for c in CAT_BINARY_COLS]
        try:
            sample_input = _synth_sample(cont_fields, cat_fields, 5, 0)
        except Exception as e:
            print("警告：内置示例生成失败：", e)

    # 3. 合成 SHAP 背景样本（不含真实患者行）
    bg_raw = bg_scaled = None
    try:
        bg_raw, bg_scaled = _synth_bg(FINAL_FEATURES, CAT_BINARY_COLS, train_means,
                                      train_std, scaler, 200, 0)
        print(f"[5] 合成背景样本: {len(bg_raw)} 条（合成，不含真实患者）")
    except Exception as e:
        print("警告：背景样本合成失败（Web 端 SHAP 将不可用）：", e)
        bg_raw = bg_scaled = None

    # 4. 组装可部署包
    pkg = {
        "meta": {
            "model_name": BEST_MODEL_NAME,
            "task": "二分类",
            "threshold": BEST_THRESHOLD,
            "classes": [0, 1],
            "class_names": {0: "无效", 1: "有效"},
            "final_features": FINAL_FEATURES,
            "cat_binary_cols": CAT_BINARY_COLS,
            "cat_multi_cols": CAT_MULTI_COLS,
            "target_col": TARGET_COL,
        },
        "model": model,
        "scaler": scaler,
        "train_means": train_means,
        "bg_raw": bg_raw,
        "bg_scaled": bg_scaled,
        "sample_input": sample_input,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[6] 打包完成: {OUT_PKL}")
    print("\n打包完成 ✅ 文件已可直接用于 Streamlit 部署。")


if __name__ == "__main__":
    build_package()
