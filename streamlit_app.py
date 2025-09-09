# streamlit_app.py (อัปเดตใหม่ พร้อมภาษาไทย+อังกฤษ)
import os, math, json, sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from locations_th import PROV_TO_DIST, DIST_TO_SUB, SUB_TO_STREET, STREET_TO_ZONE
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Setup ----------
st.set_page_config(page_title="Condo Price Predictor", page_icon="🏢", layout="wide")

FLAGS = ["is_pool_access", "is_corner", "is_high_ceiling"]  # เราจะล็อก = 0 เสมอ

NUM_FEATURES = [
    "Area_sqm", "Project_Age_notreal", "Floors", "Total_Units",
    "Launch_Month_sin", "Launch_Month_cos",
] + FLAGS  # ยังต้องอยู่ใน ALL_FEATURES เพื่อให้ตรงกับโมเดล

CAT_FEATURES = [
    "Room_Type_Base", "Province", "District", "Subdistrict", "Street", "Zone"
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
PIPELINE_FILE = "pipeline.pkl"

# ======== TwoSegmentRegressor shim for unpickle ========
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

def _ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = 0
    return d[cols]

@dataclass
class TwoSegmentRegressor:
    threshold: float
    selected_features: list
    cat_cols: list
    num_cols: list
    mass_encoder: object
    mass_model: object
    lux_encoder: object
    lux_model: object
    gate_enc: object
    gate_clf: object
    gate_cutoff: float = 0.5
    info: dict = field(default_factory=dict)

    def _predict_side_mask(self, X: pd.DataFrame):
        Xp = _ensure_columns(X, self.selected_features)
        Xg = Xp.copy()
        if len(self.cat_cols):
            Xg[self.cat_cols] = self.gate_enc.transform(Xg[self.cat_cols])
        p = self.gate_clf.predict_proba(Xg)[:, 1]
        side = (p >= self.gate_cutoff)  # True = LUX
        return side, p

    def predict(self, X: pd.DataFrame):
        Xp = _ensure_columns(X, self.selected_features)
        side, _ = self._predict_side_mask(Xp)
        yhat = np.empty(len(Xp), dtype=float)

        if (~side).any():
            Xm = Xp.loc[~side].copy()
            if len(self.cat_cols):
                Xm[self.cat_cols] = self.mass_encoder.transform(Xm[self.cat_cols])
            yhat[~side] = self.mass_model.predict(Xm)

        if side.any():
            Xl = Xp.loc[side].copy()
            if len(self.cat_cols):
                Xl[self.cat_cols] = self.lux_encoder.transform(Xl[self.cat_cols])
            yhat[side] = self.lux_model.predict(Xl)

        return yhat

    def predict_side(self, X: pd.DataFrame):
        side, prob = self._predict_side_mask(X)
        lab = np.where(side, "LUX", "MASS")
        return lab, prob

# 👇 บรรทัดนี้สำคัญ: ทำให้ pickle มองว่าคลาสอยู่ในโมดูล "main"
TwoSegmentRegressor.__module__ = "main"
# ======== end shim ========

# ---------- Helpers ----------
import re, math, os, sys
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===== Numeric columns resolver (ไม่พึ่ง NUM_ONLY ตอนประกาศ) =====
NUM_ONLY_FALLBACK = ["Area_sqm","Project_Age_notreal","Floors","Total_Units","Launch_Month_sin","Launch_Month_cos"]

def _resolve_num_cols(df_like=None):
    """คืนรายชื่อคอลัมน์ตัวเลข:
       - ถ้ามี NUM_ONLY ใน globals -> ใช้นั้น
       - ไม่งั้นใช้ fallback แล้วกรองให้เหลือเฉพาะคอลัมน์ที่มีจริง (ถ้าส่ง df_like มา)"""
    num_cols = globals().get("NUM_ONLY", NUM_ONLY_FALLBACK)
    if df_like is not None and hasattr(df_like, "columns"):
        num_cols = [c for c in num_cols if c in df_like.columns]
    return num_cols

# ===== Confidence helpers (numeric part; log1p ลด outlier) =====
def _prep_num(df, num_cols=None):
    """เตรียมคอลัมน์ตัวเลขก่อนสเกล (ลด outlier ของ Total_Units ด้วย log1p)"""
    if num_cols is None:
        num_cols = _resolve_num_cols(df)
    z = df.copy()
    for c in num_cols:
        if c not in z.columns:
            z[c] = 0.0
    z = z[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "Total_Units" in z.columns:
        z["Total_Units"] = np.log1p(z["Total_Units"])
    return z

def _robust_scale_fit_nums(X_num: pd.DataFrame):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    r = RobustScaler().fit(X_num)
    X_r = r.transform(X_num)
    s = StandardScaler().fit(X_r)
    return r, s

def _robust_scale_transform_nums(X_num: pd.DataFrame, r, s):
    return s.transform(r.transform(X_num))

def _auto_top_k(n_train: int):
    return int(np.clip(np.sqrt(max(1, n_train)), 5, 25))

def _train_similarity_distribution(Xt_scaled: np.ndarray, top_k=10):
    sim = cosine_similarity(Xt_scaled)
    np.fill_diagonal(sim, -np.inf)
    topk_mean = np.mean(np.sort(sim, axis=1)[:, -min(top_k, sim.shape[1]-1):], axis=1)
    return (topk_mean + 1.0) / 2.0

# ===== Drift report (winsorized z-score) =====
TOP_Z = 2.0
def _dimension_drift_report(X_train_all, X_input_one, num_cols=None, topn=3):
    """รายงานคอลัมน์ตัวเลขที่ต่างจาก training มากที่สุด โดย clip z-score ใน [-TOP_Z, TOP_Z]"""
    if num_cols is None:
        num_cols = _resolve_num_cols(X_train_all)
    Xt = _prep_num(X_train_all[num_cols], num_cols)
    mu = Xt.mean()
    sd = Xt.std().replace(0, 1.0)
    x = _prep_num(X_input_one[num_cols], num_cols).iloc[0]
    z = ((x - mu) / sd).clip(-TOP_Z, TOP_Z).abs().sort_values(ascending=False)
    rep = [(c, float(z[c]), float(x[c])) for c in z.index[:topn]]
    return rep

# ===== Hybrid Confidence rescale + label =====
def _rescale_and_label(conf_raw: float, low=0.20, high=0.85):
    """rescale แบบ affine ให้ช่วงใช้งานอยู่ระหว่าง ~20–85% (ปรับได้)
       คืนค่า (conf_rescaled_0_1, label, icon)"""
    cr = float(conf_raw)
    conf_rescaled = (cr - low) / (high - low)
    conf_rescaled = float(np.clip(conf_rescaled, 0.0, 1.0))
    if conf_rescaled >= 0.75:
        return conf_rescaled, "เหมือนมาก", "✅"
    elif conf_rescaled >= 0.45:
        return conf_rescaled, "ใกล้เคียง", "ℹ️"
    else:
        return conf_rescaled, "ต่างเยอะ", "⚠️"

# ===== Numeric-only confidence (percentile) =====
def numeric_only_confidence(X_input, scaler_r, scaler_s, Xt_scaled_train, dist_ref_01, top_k, num_cols=None):
    if num_cols is None:
        num_cols = _resolve_num_cols(X_input)
    x_num = _prep_num(X_input[num_cols], num_cols)
    x_scaled = _robust_scale_transform_nums(x_num, scaler_r, scaler_s)
    sim = cosine_similarity(Xt_scaled_train, x_scaled).ravel()
    k = min(top_k, len(sim))
    topk_mean = np.mean(np.sort(sim)[-k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct  # 0..1

# ===== Categorical helpers for Hybrid Confidence =====
CAT_FOR_CONF = ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]

def _fit_cat_encoder(X_train_all: pd.DataFrame, cat_cols=CAT_FOR_CONF):
    """ฟิต OneHotEncoder บนคอลัมน์หมวดหมู่ของ X_train_all แล้วคืน (encoder, X_cat_train_encoded)"""
    Xt = X_train_all.copy()
    for c in cat_cols:
        if c not in Xt.columns:
            Xt[c] = ""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(Xt[cat_cols].astype(str))
    X_cat_train = enc.transform(Xt[cat_cols].astype(str)).astype(float)
    return enc, X_cat_train

def _topk_mean_self_distribution(mat: np.ndarray, top_k: int = 10) -> np.ndarray:
    """แจกแจงอ้างอิง: mean(top-k cosine sim) ของแต่ละแถวเทียบกับ train เอง (ตัดแนวทแยง)"""
    if mat.shape[0] <= 1:
        return np.array([1.0], dtype=float)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, -np.inf)
    k = min(top_k, sim.shape[1]-1)
    topk = np.sort(sim, axis=1)[:, -k:]
    topk_mean = topk.mean(axis=1)
    return (topk_mean + 1.0) / 2.0

def cat_similarity_percentile(
    X_input: pd.DataFrame,
    enc: OneHotEncoder,
    X_cat_train: np.ndarray,
    cat_cols=CAT_FOR_CONF,
    top_k: int = 10,
) -> float:
    """ความคล้ายเชิงหมวดหมู่แบบ percentile: mean(top-k cosine sim) ของอินพุต เทียบกับ distribution ของ train"""
    x_cat = X_input.copy()
    for c in cat_cols:
        if c not in x_cat.columns:
            x_cat[c] = ""
    x_vec = enc.transform(x_cat[cat_cols].astype(str)).astype(float)
    sim = cosine_similarity(X_cat_train, x_vec).ravel()
    k = min(top_k, len(sim))
    topk_mean = np.mean(np.sort(sim)[-k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    dist_ref_01 = _topk_mean_self_distribution(X_cat_train, top_k=k)
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct  # 0..1

# ===== Normalizers / text utils =====
def _norm_obj(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())

def _unique_normalized(series: pd.Series):
    return set(series.dropna().astype(str).map(_norm_obj).unique())

def _norm_txt(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _top_counts(series, topk=5):
    vc = series.dropna().astype(str).str.strip().value_counts()
    return [(z, int(c)) for z, c in vc.head(topk).items()]

# ===== Geo chain & zone guess =====
def _filter_chain(df, province=None, district=None, subdistrict=None, street=None):
    """คืน list ของ (label, df_filtered) ตามลำดับความจำเพาะ → กว้าง"""
    steps = []
    def _match(col, val):
        return df[col].astype(str).str.strip().str.lower() == _norm_txt(val)

    if street:
        df4 = df[_match("Province", province) & _match("District", district) & _match("Subdistrict", subdistrict) & _match("Street", street)]
        steps.append(("prov+dist+sub+street", df4))
    if subdistrict:
        df3 = df[_match("Province", province) & _match("District", district) & _match("Subdistrict", subdistrict)]
        steps.append(("prov+dist+sub", df3))
    if district:
        df2 = df[_match("Province", province) & _match("District", district)]
        steps.append(("prov+dist", df2))
    if province:
        df1 = df[_match("Province", province)]
        steps.append(("prov", df1))
    return steps

def guess_zone(province, district, subdistrict, street, xtrain_df, street_to_zone=None, topk=5):
    """
    คืน (best_zone, candidates:list[(zone,count)], picked_from:str)
    - ใช้ xtrain_df['Zone'] เป็นฐานโหวต
    - ถ้าไม่เจอเลยจะ fallback ที่ street_to_zone
    """
    best_zone, candidates, picked_from = "", [], ""
    if xtrain_df is not None and "Zone" in xtrain_df.columns:
        for tag, dff in _filter_chain(xtrain_df, province, district, subdistrict, street):
            if len(dff):
                cands = _top_counts(dff["Zone"], topk=topk)
                if cands:
                    best_zone = cands[0][0]
                    candidates = cands
                    picked_from = tag
                    break
    if not best_zone and street_to_zone is not None:
        z = street_to_zone.get(street, "")
        if z:
            best_zone = z
            candidates = [(z, 0)]
            picked_from = "street_mapping"
    return best_zone, candidates, picked_from

# ===== Misc =====
def month_to_sin_cos(m: int):
    rad = 2 * math.pi * (m - 1) / 12.0
    return math.sin(rad), math.cos(rad)

def safe_float(x, default=0.0):
    try: return float(x)
    except: return float(default)

def ensure_columns(df: pd.DataFrame, cols: list, fill_value_map: dict = None) -> pd.DataFrame:
    """ทำให้ df มีคอลัมน์ตรงกับ cols ทุกตัว; ถ้าไม่มีให้เติม และเรียงคอลัมน์ให้เหมือนกัน"""
    df = df.copy()
    fill_value_map = fill_value_map or {}
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value_map.get(c, 0)
    return df[cols]

# ===== UI helper =====
def flexible_selectbox(label, options):
    """เลือกจาก list หรือพิมพ์เองได้ (return ค่าที่เลือก/พิม)"""
    extended_options = options + ["อื่น ๆ (พิมพ์เอง)"]
    choice = st.selectbox(label, extended_options)
    if choice == "อื่น ๆ (พิมพ์เอง)":
        manual_value = st.text_input(f"กรุณาพิมพ์ {label} ที่ต้องการ")
        return manual_value.strip()
    else:
        return choice

# ===== Encoding / confidence backfills =====
def _which_side(pipeline, X_one_row):
    """บอกว่าพยากรณ์แถวนี้ใช้ MASS หรือ LUX (fallback ถ้าไม่มี predict_side)"""
    if hasattr(pipeline, "predict_side"):
        try:
            lab, _ = pipeline.predict_side(pd.DataFrame([X_one_row]))
            if len(lab): return lab[0]
        except Exception:
            pass
    return "LUX" if float(X_one_row.get("Area_sqm", 0)) > 250 else "MASS"

def _encode_like_model(pipeline, X_df, cat_cols, side):
    """พยายามเข้ารหัสแบบเดียวกับโมเดล; ถ้าไม่มี ให้ one-hot ชั่วคราว (ใช้เฉพาะคำนวณความมั่นใจ)"""
    X_df = X_df.copy()
    if side == "LUX" and hasattr(pipeline, "lux_encoder") and pipeline.lux_encoder is not None:
        X_df[cat_cols] = pipeline.lux_encoder.transform(X_df[cat_cols])
        return X_df, "model"
    if side == "MASS" and hasattr(pipeline, "mass_encoder") and pipeline.mass_encoder is not None:
        X_df[cat_cols] = pipeline.mass_encoder.transform(X_df[cat_cols])
        return X_df, "model"
    pre = ColumnTransformer([
        ("num", "passthrough", [c for c in X_df.columns if c not in cat_cols]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    enc = Pipeline([("pre", pre)])
    X_arr = enc.fit_transform(X_df)   # ถ้าต้องการให้คอลัมน์ตรง ควร fit บน train+input ที่ concat ภายนอก
    X_encoded = pd.DataFrame(X_arr)
    return X_encoded, "onehot"

def compute_confidence_robust(pipeline, X_train_all, X_input_one, all_cols, cat_cols, top_k=5):
    """คำนวณความคล้ายด้วย cosine บน representation ที่เข้ารหัสแล้ว (fallback one-hot ได้)"""
    X_train_used = X_train_all.reindex(columns=all_cols).copy()
    X_input = X_input_one.reindex(columns=all_cols).copy()
    side = _which_side(pipeline, X_input.iloc[0].to_dict())

    X_train_enc, mode1 = _encode_like_model(pipeline, X_train_used, cat_cols, side)
    if mode1 == "model":
        X_input_enc, _ = _encode_like_model(pipeline, X_input, cat_cols, side)
    else:
        combo = pd.concat([X_train_used, X_input], axis=0)
        X_combo_enc, _ = _encode_like_model(pipeline, combo, cat_cols, side)
        X_train_enc = X_combo_enc.iloc[:-1, :].reset_index(drop=True)
        X_input_enc = X_combo_enc.iloc[-1:, :].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_new_scaled = scaler.transform(X_input_enc)

    sim = cosine_similarity(X_train_scaled, X_new_scaled).ravel()
    k = min(top_k, len(sim))
    conf = float(np.mean(np.sort(sim)[-k:]))
    return conf


# ---------- โหลด X_train (สำหรับ Confidence) ----------
# ---------- Load model ----------
try:
    import two_segment
    sys.modules['main'] = two_segment   # ให้ pickle หา class TwoSegmentRegressor เจอ
except Exception:
    pass

PIPELINE_FILE = "pipeline.pkl"

if not os.path.exists(PIPELINE_FILE):
    st.error(f"ไม่พบไฟล์ {PIPELINE_FILE} — กรุณาวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับสคริปต์")
    st.stop()

try:
    import joblib
    pipeline = joblib.load(PIPELINE_FILE)
    st.sidebar.success("โหลด pipeline.pkl สำเร็จ ✅")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ===== Confidence (numeric-only percentile) setup =====
NUM_ONLY = ["Area_sqm","Project_Age_notreal","Floors","Total_Units","Launch_Month_sin","Launch_Month_cos"]
# ---------- Load y_train for Conformal ----------
y_train_all = None
try:
    if os.path.exists("y_train.pkl"):
        y_train_all = joblib.load("y_train.pkl")
    elif os.path.exists("y_train.npy"):
        y_train_all = np.load("y_train.npy")
except Exception as e:
    st.sidebar.warning(f"โหลด y_train ไม่สำเร็จ: {e}")
    y_train_all = None

# ตรวจความสอดคล้องระหว่าง X_train_all และ y_train_all
def _align_Xy_for_conformal(Xt: pd.DataFrame, y):
    import numpy as np, pandas as pd
    if Xt is None or y is None:
        return None, None
    y = pd.Series(y)
    # ถ้า index ตรงกัน ใช้ index ร่วม
    if isinstance(y.index, type(Xt.index)) and (Xt.index.equals(y.index)):
        return Xt, y
    # ถ้าไม่ตรง: ใช้ความยาวขั้นต่ำ (กัน shape mismatch)
    n = min(len(Xt), len(y))
    if n == 0:
        return None, None
    Xt2 = Xt.iloc[:n].copy()
    y2 = y.iloc[:n].copy().reset_index(drop=True)
    Xt2.reset_index(drop=True, inplace=True)
    return Xt2, y2

Xt_for_conf, y_for_conf = _align_Xy_for_conformal(X_train_all, y_train_all)


def _fit_numeric_scaler(X_train_all, num_cols=NUM_ONLY):
    Xt = X_train_all.copy()
    # เติมคอลัมน์ที่ขาด → 0 (กัน KeyError)
    for c in num_cols:
        if c not in Xt.columns:
            Xt[c] = 0.0
    Xt = Xt[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler().fit(Xt)      # fit เฉพาะ train
    Xt_scaled = scaler.transform(Xt)
    return scaler, Xt_scaled

def _train_similarity_distribution(Xt_scaled, top_k=10):
    # cosine กับตัวเอง แล้วตัดค่าทะแยง (ไม่เทียบตัวเอง)
    sim = cosine_similarity(Xt_scaled)
    np.fill_diagonal(sim, -np.inf)
    # ค่าเฉลี่ย top-k เพื่อนบ้านที่ใกล้สุดของแต่ละแถวใน train
    topk_mean = np.mean(np.sort(sim, axis=1)[:, -top_k:], axis=1)
    # map [-1,1] → [0,1]
    topk_mean_01 = (topk_mean + 1.0) / 2.0
    return topk_mean_01

def confidence_numeric_percentile(X_input, scaler, Xt_scaled_train, dist_ref_01, num_cols=NUM_ONLY, top_k=10):
    x = X_input.copy()
    for c in num_cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_scaled = scaler.transform(x)
    sim = cosine_similarity(Xt_scaled_train, x_scaled).ravel()
    topk_mean = np.mean(np.sort(sim)[-top_k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    # เปอร์เซ็นไทล์เมื่อเทียบกับ distribution ของ train
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct

def _robust_scale_fit(X: pd.DataFrame):
    r = RobustScaler().fit(X)
    X_r = r.transform(X)
    s = StandardScaler().fit(X_r)
    return r, s

def _robust_scale_transform(X: pd.DataFrame, r, s):
    return s.transform(r.transform(X))

def _auto_top_k(n_train: int):
    import math
    return int(np.clip(math.sqrt(max(1, n_train)), 5, 25))

# ===== Numeric-only percentile setup (ปรับให้ใช้ log1p กับ Total_Units) =====
# ===== Numeric-only percentile setup =====
conf_ready = False
if X_train_all is not None and isinstance(X_train_all, pd.DataFrame) and len(X_train_all) > 0:
    num_cols = _resolve_num_cols(X_train_all)
    Xt_num = _prep_num(X_train_all[num_cols], num_cols)
    r_scaler, s_scaler = _robust_scale_fit_nums(Xt_num)
    Xt_scaled_train = _robust_scale_transform_nums(Xt_num, r_scaler, s_scaler)
    TOPK_REF = _auto_top_k(len(Xt_scaled_train))
    dist_ref_01 = _train_similarity_distribution(Xt_scaled_train, top_k=TOPK_REF)

    # Categorical encoder สำหรับ hybrid (เหมือนเดิม)
    # Categorical encoder สำหรับ Hybrid
    cat_enc, X_cat_train = _fit_cat_encoder(X_train_all, CAT_FOR_CONF)


    conf_ready = True



def confidence_numeric_percentile(X_input, r_scaler, s_scaler, Xt_scaled_train, dist_ref_01,
                                  num_cols=NUM_ONLY, top_k=10):
    x = X_input.copy()
    for c in num_cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_scaled = _robust_scale_transform(x, r_scaler, s_scaler)

    sim = cosine_similarity(Xt_scaled_train, x_scaled).ravel()
    k = min(top_k, len(sim))
    topk_mean = np.mean(np.sort(sim)[-k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct
from sklearn.preprocessing import OneHotEncoder

CAT_FOR_CONF = ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]

def _fit_cat_encoder(X_train_all, cat_cols=CAT_FOR_CONF):
    Xt = X_train_all.copy()
    for c in cat_cols:
        if c not in Xt.columns: Xt[c] = ""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(Xt[cat_cols].astype(str))
    X_cat = enc.transform(Xt[cat_cols].astype(str))
    return enc, X_cat


def cat_similarity_percentile(X_input, enc, X_cat_train, cat_cols=CAT_FOR_CONF, top_k=10):
    xi = X_input.copy()
    for c in cat_cols:
        if c not in xi.columns: xi[c] = ""
    Xi_cat = enc.transform(xi[cat_cols].astype(str))
    sim = cosine_similarity(X_cat_train, Xi_cat).ravel()
    k = min(top_k, len(sim))
    topk_mean = np.mean(np.sort(sim)[-k:])
    conf_01 = (topk_mean + 1.0) / 2.0

    sim_train = cosine_similarity(X_cat_train)
    np.fill_diagonal(sim_train, -np.inf)
    topk_mean_train = np.mean(np.sort(sim_train, axis=1)[:, -k:], axis=1)
    topk_mean_train_01 = (topk_mean_train + 1.0) / 2.0
    pct = float((topk_mean_train_01 <= conf_01).mean())
    return pct

# สร้าง encoder สำหรับ categorical
if conf_ready:
    cat_enc, X_cat_train = _fit_cat_encoder(X_train_all, CAT_FOR_CONF)
# ใช้ Winsorize (clip) z-score เพื่อลดผล outlier ในรายงาน
TOP_Z = 2.0  # ปรับได้ 1.5–3.0

def _dimension_drift_report(X_train_all, X_input_one, num_cols=NUM_ONLY, topn=3):
    """รายงานคอลัมน์ตัวเลขที่ต่างจาก training มากที่สุด โดย clip z-score ใน [-TOP_Z, TOP_Z]"""
    rep = []
    Xt = X_train_all[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mu = Xt.mean()
    sd = Xt.std().replace(0, 1.0)

    x = X_input_one[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).iloc[0]

    # z-score แล้ว clip ลดอิทธิพลค่าที่สุดโต่ง
    z_raw = (x - mu) / sd
    z = z_raw.clip(-TOP_Z, TOP_Z).abs().sort_values(ascending=False)

    for c in z.index[:topn]:
        rep.append((c, float(z[c]), float(x[c])))
    return rep

# ---------- Conformal Calibration ----------
def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    if n <= 0:
        return float("nan")
    # split-conformal correction: (1 - alpha)*(1 + 1/n)
    q = np.quantile(residuals, min(1.0, (1 - alpha) * (1 + 1.0 / n)), method="higher")
    return float(q)

def fit_conformal_from_calib(pipeline, X_train_df, y_train_arr, calib_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X_train_df))
    rng.shuffle(idx)
    n_cal = max(50, int(len(idx) * calib_frac))
    calib_idx = idx[:n_cal]
    Xc = X_train_df.iloc[calib_idx].copy()
    yc = np.asarray(y_train_arr)[calib_idx]

    # ให้คอลัมน์ครบเหมือนทำนายจริง
    Xc = ensure_columns(Xc, ALL_FEATURES, fill_value_map=None)

    yhat_c = np.ravel(pipeline.predict(Xc))
    resid = np.abs(yc - yhat_c)

    return {
        "q90": _conformal_quantile(resid, 0.10),
        "q95": _conformal_quantile(resid, 0.05),
        "n_calib": int(len(resid)),
    }

conformal_ready, conformal_info = False, None
try:
    if Xt_for_conf is not None and y_for_conf is not None:
        Xt_full = ensure_columns(Xt_for_conf, ALL_FEATURES, fill_value_map=None)
        conformal_info = fit_conformal_from_calib(pipeline, Xt_full, y_for_conf, calib_frac=0.2, seed=42)
        conformal_ready = True
        st.sidebar.success(
            f"Conformal ready ✅ | calib_n={conformal_info['n_calib']}, "
            f"q90={conformal_info['q90']:.3f}, q95={conformal_info['q95']:.3f}"
        )
    else:
        st.sidebar.warning("⚠️ Conformal ยังไม่พร้อม (Xt/y ว่างหรือยาวไม่พอ)")
except Exception as e:
    st.sidebar.warning(f"ตั้งค่า Conformal ไม่สำเร็จ: {e}")
# ===== Categorical encoder + similarity for Hybrid Confidence =====
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# คอลัมน์หมวดหมู่ที่ใช้คำนวณความคล้ายเชิงหมวดหมู่
CAT_FOR_CONF = ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]

def _fit_cat_encoder(X_train_all: pd.DataFrame, cat_cols=CAT_FOR_CONF):
    """ฟิต OneHotEncoder บนคอลัมน์หมวดหมู่จาก X_train_all และคืน (encoder, X_cat_train_encoded)"""
    Xt = X_train_all.copy()
    # เติมคอลัมน์ที่หายไปด้วยค่าว่าง เพื่อกัน KeyError
    for c in cat_cols:
        if c not in Xt.columns:
            Xt[c] = ""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(Xt[cat_cols].astype(str))
    X_cat_train = enc.transform(Xt[cat_cols].astype(str)).astype(float)
    return enc, X_cat_train

def _topk_mean_self_distribution(mat: np.ndarray, top_k: int = 10) -> np.ndarray:
    """แจกแจงอ้างอิง: ค่าเฉลี่ย similarity ของ top-k เพื่อนบ้านของแต่ละแถวเทียบกับทั้ง train (ตัดทแยง)"""
    if mat.shape[0] <= 1:
        return np.array([1.0], dtype=float)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, -np.inf)
    topk = np.sort(sim, axis=1)[:, -min(top_k, sim.shape[1]-1):]
    topk_mean = topk.mean(axis=1)
    topk_mean_01 = (topk_mean + 1.0) / 2.0
    return topk_mean_01

def cat_similarity_percentile(
    X_input: pd.DataFrame,
    enc: OneHotEncoder,
    X_cat_train: np.ndarray,
    cat_cols=CAT_FOR_CONF,
    top_k: int = 10,
) -> float:
    """ความคล้ายเชิงหมวดหมู่แบบ percentile: เปรียบเทียบค่าเฉลี่ย top-k similarity ของ input กับ distribution ของ train"""
    # เข้ารหัสอินพุต
    x_cat = X_input.copy()
    for c in cat_cols:
        if c not in x_cat.columns:
            x_cat[c] = ""
    x_vec = enc.transform(x_cat[cat_cols].astype(str)).astype(float)
    # similarity กับ train
    sim = cosine_similarity(X_cat_train, x_vec).ravel()
    k = min(top_k, len(sim))
    topk_mean = np.mean(np.sort(sim)[-k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    # อ้างอิง distribution จาก train self-sim
    dist_ref_01 = _topk_mean_self_distribution(X_cat_train, top_k=k)
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct  # 0..1



# ---------- UI ----------
st.title("🏢 Condo Price Predictor")
st.caption("กรอกข้อมูล → ทำนายราคาขาย (ล้านบาท) และราคาต่อตารางเมตร (บาท/ตร.ม.)")

col1, col2, col3 = st.columns(3)
with col1:
    area_input = st.text_input("พื้นที่ (ตร.ม.) — Area_sqm", value="30")
    try: area = float(area_input)
    except: area = 0.0

    floors_input = st.text_input("ชั้นอาคาร — Floors", value="8")
    try: floors = int(floors_input)
    except: floors = 1

with col2:
    age_input = st.text_input("อายุโครงการ (ปี) — Project_Age", value="0")
    try: age = float(age_input)
    except: age = 0.0

    total_units_input = st.text_input("จำนวนยูนิตทั้งหมด — Total_Units", value="300")
    try: total_units = int(total_units_input)
    except: total_units = 100

with col3:
    month = st.selectbox("เดือนเปิดตัว — Launch Month", options=list(range(1,13)), index=0)
    m_sin, m_cos = month_to_sin_cos(month)

# 🏢 ใช้กับจังหวัด-อำเภอ-ตำบล-ถนน
province = flexible_selectbox("จังหวัด - Province", sorted(PROV_TO_DIST))
district = flexible_selectbox("เขต/อำเภอ - District", PROV_TO_DIST.get(province, []))
subdistrict = flexible_selectbox("แขวง/ตำบล - Subdistrict", DIST_TO_SUB.get(district, []))
street = flexible_selectbox("ถนน - Street", SUB_TO_STREET.get(subdistrict, []))


# 🌐 Zone (auto) — ใช้ทั้ง Province/District/Subdistrict/Street + โหวตจาก X_train
if X_train_all is not None and isinstance(X_train_all, pd.DataFrame):
    try:
        # บังคับคอลัมน์ให้ครบก่อน (กัน key error)
        needed = ["Province","District","Subdistrict","Street","Zone"]
        xtrain_geo = X_train_all.copy()
        for c in needed:
            if c not in xtrain_geo.columns:
                xtrain_geo[c] = ""

        zone_guess, zone_cands, picked_from = guess_zone(
            province=province,
            district=district,
            subdistrict=subdistrict,
            street=street,
            xtrain_df=xtrain_geo,
            street_to_zone=STREET_TO_ZONE,
            topk=6
        )

        # ทำตัวเลือก: เอา best ขึ้นก่อน แล้วตามด้วย candidates อื่น ๆ และ "อื่น ๆ (พิมพ์เอง)"
        options = []
        if zone_guess: options.append(zone_guess)
        for z, _cnt in zone_cands:
            if z and z not in options:
                options.append(z)
        # เพิ่ม fallback จาก mapping ถนนถ้ายังไม่มี
        z_map = STREET_TO_ZONE.get(street, "")
        if z_map and z_map not in options:
            options.append(z_map)
        options.append("อื่น ๆ (พิมพ์เอง)")

        zone_choice = st.selectbox("Zone (auto, ปรับได้)", options=options, index=0)
        if zone_choice == "อื่น ๆ (พิมพ์เอง)":
            zone = st.text_input("พิมพ์ Zone เอง", value=zone_guess or z_map or "")
        else:
            zone = zone_choice

        # แสดง hint แหล่งที่มาและอันดับโหวต
        with st.expander("ดูเหตุผลการเดา Zone", expanded=False):
            st.write(f"picked_from: **{picked_from or 'fallback'}**")
            if zone_cands:
                st.write("Top zones ใกล้เคียงจากข้อมูลฝึก:")
                st.table(pd.DataFrame(zone_cands, columns=["Zone","Count"]))
    except Exception as e:
        st.warning(f"เดา Zone อัตโนมัติไม่ได้: {e}")
        zone = st.text_input("Zone (manual)", value=STREET_TO_ZONE.get(street,""))
else:
    # ถ้ายังไม่มี X_train → fallback ตามถนนเหมือนเดิม แต่ให้แก้ไขได้
    zone = st.text_input("Zone (auto from street / editable)", value=STREET_TO_ZONE.get(street, ""))








room_type_base = st.selectbox("ประเภทห้อง — Room_Type", options = [
    'STUDIO', '2BED', '3BED', '1BED', '1BED_PLUS', 'PENTHOUSE', '2BED_DUPLEX',
    '1BED_DUPLEX', 'DUPLEX_OTHER', '4BED', 'POOL_VILLA', '4BED_PENTHOUSE',
    '3BED_DUPLEX', '1BED_LOFT', '3BED_TRIPLEX', '3BED_PENTHOUSE', '4BED_DUPLEX',
    '5BED_DUPLEX', '2BED_PLUS', 'PENTHOUSE_DUPLEX', 'Pool Access(เชื่อมสระว่ายน้ำ)',
    '5BED', 'MOFF-Design', '25BED', 'LOFT_OTHER', '2BED_PENTHOUSE', 'SHOP',
    '1BED_PLUS_LOFT', '2BED_LOFT', 'Stuio vertiplex', '3BED_PLUS', '3BED_PLUS_DUPLEX',
    '3BED_LOFT', '4BED_LOFT', 'DUO', '1BED_TRIPLEX', '1BED_PLUS_TRIPLEX',
    '2BED_TRIPLEX', 'Simplex'])



# ✅ สร้าง row ก่อน (ไม่มีอินพุตสำหรับ FLAGS)
row = {
    "Area_sqm": area,
    "Project_Age_notreal": age,
    "Floors": floors,
    "Total_Units": total_units,
    "Launch_Month_sin": m_sin,
    "Launch_Month_cos": m_cos,
    "Province": province,
    "District": district,
    "Subdistrict": subdistrict,
    "Street": street,
    "Zone": zone,
    "Room_Type_Base": room_type_base,
    "is_pool_access": 0,
    "is_corner": 0,
    "is_high_ceiling": 0,
}


# ✅ แล้วค่อยสร้าง DataFrame X
X = pd.DataFrame([row], columns=ALL_FEATURES)

# ✅ ค่อยมาเช็ค unseen values (แบบ normalize)
unseen_cols = []
if 'X_train_all' in globals() and isinstance(X_train_all, pd.DataFrame) and X_train_all is not None:
    for col in ALL_FEATURES:
        if col not in X.columns or col not in X_train_all.columns:
            continue
        user_value = X[col].iloc[0]
        if X[col].dtype == 'object' or isinstance(user_value, str):
            norm_user = _norm_obj(user_value)
            train_uni = _unique_normalized(X_train_all[col])
            if norm_user not in train_uni:
                unseen_cols.append(col)

if unseen_cols:
    st.warning(f"⚠️ ค่าต่อไปนี้ไม่เคยปรากฏในการฝึกโมเดล (หลัง normalize): {', '.join(unseen_cols)}")

# ---------- Predict ----------
if st.button("Predict Price (ล้านบาท)"):
    try:
        # ===== ทำนายราคา =====
        y_pred = pipeline.predict(X)
        pred_val = float(np.ravel(y_pred)[0])
        st.metric("ราคาคาดการณ์ (ล้านบาท)", f"{pred_val:.3f}")

        price_per_sqm = (pred_val * 1_000_000.0) / max(1.0, safe_float(area, 1.0))
        st.metric("ราคาต่อตารางเมตร (บาท/ตร.ม.)", f"{price_per_sqm:,.0f}")

        # ===== แสดงฝั่งที่ router ส่งไป + ความเชื่อมั่นของ gate =====
        try:
            side_label, side_prob = pipeline.predict_side(X)
            side_txt = side_label[0]
            st.caption(f"Router side: **{side_txt}**  (P[LUX]={side_prob[0]:.2f})")
        except Exception:
            pass

        # ===== Conformal Prediction Intervals =====
        if conformal_ready and (conformal_info is not None):
            q90, q95 = conformal_info["q90"], conformal_info["q95"]
            pi90 = (max(0.0, pred_val - q90), max(0.0, pred_val + q90))
            pi95 = (max(0.0, pred_val - q95), max(0.0, pred_val + q95))

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Prediction Interval 90%")
                st.success(f"[{pi90[0]:.3f} , {pi90[1]:.3f}] ล้านบ.")
            with c2:
                st.caption("Prediction Interval 95%")
                st.info(f"[{pi95[0]:.3f} , {pi95[1]:.3f}] ล้านบ.")
        else:
            st.warning("⚠️ ไม่มีคาลิเบรชันสำหรับ Conformal → ยังไม่แสดงช่วงคาดการณ์ (PI)")

        # ===== Hybrid Confidence (คำนวณดิบ → rescale → label) =====
        if conf_ready:
            try:
                # numeric part
                num_conf = confidence_numeric_percentile(
                    X, r_scaler, s_scaler, Xt_scaled_train, dist_ref_01,
                    NUM_ONLY, top_k=_auto_top_k(len(Xt_scaled_train))
                )
                # categorical part
                cat_conf = cat_similarity_percentile(
                X, cat_enc, X_cat_train, CAT_FOR_CONF, top_k=_auto_top_k(len(X_cat_train))
                )

                # ผสม (เลือก rate ได้: 0.2–0.4)
                HYBRID_ALPHA = 0.25
                conf_raw = HYBRID_ALPHA * num_conf + (1 - HYBRID_ALPHA) * cat_conf

                # penalty เบา ๆ เมื่อพบ unseen categories
                cat_miss = []
                for c in ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]:
                    if c in X.columns and c in X_train_all.columns:
                        if _norm_obj(X.iloc[0][c]) not in _unique_normalized(X_train_all[c]):
                            cat_miss.append(c)
                if cat_miss:
                    conf_raw *= 0.9  # -10%

                # rescale + label
                conf_rescaled, conf_label, conf_icon = _rescale_and_label(conf_raw, low=0.20, high=0.85)
                st.metric("ความมั่นใจของโมเดล (Hybrid Confidence)", f"{conf_rescaled*100:.1f} %", help=f"Label: {conf_label}")

                # แสดง label เด่น ๆ
                st.info(f"{conf_icon} ระดับความคล้าย: **{conf_label}**  (raw={conf_raw*100:.1f}%)")

                # (option) โชว์ numeric-only ไว้เทียบ
                with st.expander("ดูค่า numeric-only / รายละเอียด (กดเพื่อเปิด)", expanded=False):
                    num_only_pct = numeric_only_confidence(
                        X, r_scaler, s_scaler, Xt_scaled_train, dist_ref_01,
                        _auto_top_k(len(Xt_scaled_train))
                    )
                    st.write(f"Numeric-only confidence: **{num_only_pct*100:.1f}%**")
                    st.caption(f"(alpha={HYBRID_ALPHA:.2f}, hybrid_raw={conf_raw*100:.1f}%)")

                # Diagnostics เมื่อ “ต่างเยอะ”
                if conf_label == "ต่างเยอะ":
                    with st.expander("🔎 ทำไมความมั่นใจต่ำ? (รายละเอียด)", expanded=False):
                        dr = _dimension_drift_report(X_train_all, X, NUM_ONLY, topn=3)
                        if dr:
                            st.write("คอลัมน์ตัวเลขที่ต่างจาก training มากที่สุด (|z|-score, clipped):")
                            st.table(pd.DataFrame(dr, columns=["Column","|z|","Input value"]))
                        if cat_miss:
                            st.write("หมวดหมู่ที่ไม่เคยพบใน training (หลัง normalize): ", ", ".join(cat_miss))

            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ confidence ได้: {e}")
        else:
            st.warning("⚠️ ไม่พบ X_train.pkl หรือข้อมูล train ไม่พร้อม — จึงไม่แสดง Confidence")

    except Exception as e:
        st.error(f"ทำนายไม่สำเร็จ: {e}")

























































