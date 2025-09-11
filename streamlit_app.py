# =========================
# streamlit_app.py (FULL, Hybrid Confidence v2)
# =========================
import os, sys, math, json, warnings, re
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from locations_th import PROV_TO_DIST, DIST_TO_SUB, SUB_TO_STREET, STREET_TO_ZONE
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field

# ---------- Setup ----------
st.set_page_config(page_title="Condo Price Predictor", page_icon="🏢", layout="wide")

FLAGS = ["is_pool_access", "is_corner", "is_high_ceiling"]  # จะล็อก = 0 เสมอ
NUM_FEATURES = [
    "Area_sqm", "Project_Age_notreal", "Floors", "Total_Units",
    "Launch_Month_sin", "Launch_Month_cos",
] + FLAGS
CAT_FEATURES = ["Room_Type_Base", "Province", "District", "Subdistrict", "Street", "Zone"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

PIPELINE_FILE = "pipeline.pkl"
XTRAIN_FILE   = "X_train.pkl"
YTRAIN_FILE   = "y_train.pkl"

# =========================================
# TwoSegmentRegressor shim (for unpickle)
# =========================================

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

# ให้ pickle หา class ในโมดูล main
TwoSegmentRegressor.__module__ = "main"
sys.modules['main'] = sys.modules[__name__]

# ==================
# ----- Helpers -----
# ==================

# Numeric resolver (กันกรณียังไม่ประกาศ NUM_ONLY)
NUM_ONLY_FALLBACK = ["Area_sqm","Project_Age_notreal","Floors","Total_Units","Launch_Month_sin","Launch_Month_cos"]
def _resolve_num_cols(df_like=None):
    num_cols = globals().get("NUM_ONLY", NUM_ONLY_FALLBACK)
    if df_like is not None and hasattr(df_like, "columns"):
        num_cols = [c for c in num_cols if c in df_like.columns]
    return num_cols

def _prep_num(df, num_cols=None):
    """เตรียม numeric + log1p(Total_Units)"""
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
    r = RobustScaler().fit(X_num)
    X_r = r.transform(X_num)
    s = StandardScaler().fit(X_r)
    return r, s

def _robust_scale_transform_nums(X_num: pd.DataFrame, r, s):
    return s.transform(r.transform(X_num))

def _auto_top_k(n_train: int):
    return int(np.clip(np.sqrt(max(1, n_train)), 5, 25))

# Drift report (winsorized z-score)
TOP_Z = 2.0

def _dimension_drift_report(X_train_all, X_input_one, num_cols=None, topn=3):
    if num_cols is None:
        num_cols = _resolve_num_cols(X_train_all)
    Xt = _prep_num(X_train_all[num_cols], num_cols)
    mu = Xt.mean()
    sd = Xt.std().replace(0, 1.0)
    x = _prep_num(X_input_one[num_cols], num_cols).iloc[0]
    z = ((x - mu) / sd).clip(-TOP_Z, TOP_Z).abs().sort_values(ascending=False)
    return [(c, float(z[c]), float(x[c])) for c in z.index[:topn]]

# Hybrid rescale + label

def _rescale_and_label(conf_raw: float, low=0.20, high=0.85):
    cr = float(conf_raw)
    conf_rescaled = (cr - low) / (high - low)
    conf_rescaled = float(np.clip(conf_rescaled, 0.0, 1.0))
    if conf_rescaled >= 0.75: return conf_rescaled, "เหมือนมาก", "✅"
    if conf_rescaled >= 0.45: return conf_rescaled, "ใกล้เคียง", "ℹ️"
    return conf_rescaled, "ต่างเยอะ", "⚠️"

# Normalizers / text utils

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

# Geo chain & zone guess

def _filter_chain(df, province=None, district=None, subdistrict=None, street=None):
    steps = []
    def _match(col, val): return df[col].astype(str).str.strip().str.lower() == _norm_txt(val)
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
    best_zone, candidates, picked_from = "", [], ""
    if xtrain_df is not None and "Zone" in xtrain_df.columns:
        for tag, dff in _filter_chain(xtrain_df, province, district, subdistrict, street):
            if len(dff):
                cands = _top_counts(dff["Zone"], topk=topk)
                if cands:
                    best_zone = cands[0][0]; candidates = cands; picked_from = tag
                    break
    if not best_zone and street_to_zone is not None:
        z = street_to_zone.get(street, "")
        if z:
            best_zone = z; candidates = [(z, 0)]; picked_from = "street_mapping"
    return best_zone, candidates, picked_from

# Misc

def month_to_sin_cos(m: int):
    rad = 2 * math.pi * (m - 1) / 12.0
    return math.sin(rad), math.cos(rad)

def safe_float(x, default=0.0):
    try: return float(x)
    except: return float(default)

def ensure_columns(df: pd.DataFrame, cols: list, fill_value_map: dict = None) -> pd.DataFrame:
    df = df.copy()
    fill_value_map = fill_value_map or {}
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value_map.get(c, 0)
    return df[cols]

def flexible_selectbox(label, options):
    extended_options = options + ["อื่น ๆ (พิมพ์เอง)"]
    choice = st.selectbox(label, extended_options)
    if choice == "อื่น ๆ (พิมพ์เอง)":
        manual_value = st.text_input(f"กรุณาพิมพ์ {label} ที่ต้องการ")
        return manual_value.strip()
    else:
        return choice

# Router helpers

def _which_side(pipeline, X_one_row):
    if hasattr(pipeline, "predict_side"):
        try:
            lab, _ = pipeline.predict_side(pd.DataFrame([X_one_row]))
            if len(lab): return lab[0]
        except Exception:
            pass
    return "LUX" if float(X_one_row.get("Area_sqm", 0)) > 250 else "MASS"

# ===== New: Similarity-based Confidence (RBF+Categorical) =====
CONF_PARAMS = {
    "k_num": 50,      # k เพื่อนบ้านสำหรับฝั่งตัวเลข
    "tau": None,      # ถ้า None จะใช้ median(d^2) ของ k เพื่อนบ้าน
    "wN": 0.6,        # น้ำหนักฝั่งตัวเลข
    "wC": 0.4,        # น้ำหนักฝั่งหมวดหมู่ (wN+wC=1)
    "alpha": 0.7,     # โทษ unseen ต่อคอลัมน์ (1-alpha = match score เมื่อ unseen)
    "m": 20.0,        # smoothing ของ coverage (จำนวนเคสขั้นต่ำก่อนเชื่อค่า)
    "beta": 1.0,      # โทษ unseen รวมตามสัดส่วนคอลัมน์ที่ unseen
    "lam": 0.05,      # ฐานความมั่นใจขั้นต่ำ
    "gamma": 2.0,     # ความโค้ง (ยิ่งมากยิ่งแยกแรง)
    "cmin": 0.05,     # clip ต่ำสุดหลังคูณ base_conf
    "cmax": 0.99,     # clip สูงสุด
    "base_conf": 0.80 # ฐานความมั่นใจโดยรวมของโมเดล
}

CAT_FOR_CONF = ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]

def _rbf_numeric_similarity(x_scaled: np.ndarray, Z_train: np.ndarray, k=50, tau=None):
    k = int(min(max(1, k), len(Z_train)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Z_train)
    dists, _ = nn.kneighbors(x_scaled.reshape(1,-1), return_distance=True)
    d2 = (dists[0] ** 2)
    if tau is None:
        med = np.median(d2)
        tau = med if med > 0 else (np.mean(d2) + 1e-9)
    return float(np.mean(np.exp(-d2 / tau)))

def _build_counts_per_feature(X_train_all: pd.DataFrame, cat_cols: list):
    counts = []
    Xc = X_train_all.copy()
    for c in cat_cols:
        if c not in Xc.columns:
            Xc[c] = ""
        vc = Xc[c].astype(str).str.strip().value_counts(dropna=False)
        counts.append(vc.to_dict())
    return counts

def _categorical_similarity(x_row: pd.Series, counts_per_feature: list, cat_cols: list, alpha=0.7, m=20.0, wC=None):
    values = [str(x_row.get(c, "")).strip() for c in cat_cols]
    q = len(cat_cols)
    if q == 0:
        return 1.0, 0.0
    if wC is None:
        wC = np.ones(q) / q

    s_total = 0.0
    unseen = 0
    for j, (val, cnts) in enumerate(zip(values, counts_per_feature)):
        n = cnts.get(val, 0)
        seen = n > 0
        c = n / (n + m) if (n + m) > 0 else 0.0
        match = 1.0 if seen else (1.0 - alpha)
        s_j = match * c
        s_total += float(wC[j]) * s_j
        if not seen:
            unseen += 1

    unseen_frac = unseen / q
    return float(s_total), float(unseen_frac)

def _combine_confidence(S_num, S_cat, unseen_frac, params: dict):
    wN, wC = params["wN"], params["wC"]
    S = wN * S_num + wC * S_cat
    S_tilde = S * math.exp(-params["beta"] * unseen_frac)
    conf = (params["lam"] + (1 - params["lam"]) * S_tilde) ** params["gamma"]
    conf = conf * params["base_conf"]
    return float(np.clip(conf, params["cmin"], params["cmax"])), float(S_tilde)

# ==========================
# Load model & training data
# ==========================
if not os.path.exists(PIPELINE_FILE):
    st.error(f"ไม่พบไฟล์ {PIPELINE_FILE} — กรุณาวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับสคริปต์")
    st.stop()

try:
    pipeline = joblib.load(PIPELINE_FILE)
    st.sidebar.success("โหลด pipeline.pkl สำเร็จ ✅")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

X_train_all = None
try:
    if os.path.exists(XTRAIN_FILE):
        X_train_all = joblib.load(XTRAIN_FILE)
        if not isinstance(X_train_all, pd.DataFrame):
            X_train_all = pd.DataFrame(X_train_all)
        st.sidebar.info(f"X_train: {X_train_all.shape}")
    else:
        st.sidebar.warning("⚠️ ไม่พบ X_train.pkl — บางฟีเจอร์ (Zone guess / Confidence) จะจำกัด")
except Exception as e:
    st.sidebar.warning(f"โหลด X_train ไม่สำเร็จ: {e}")
    X_train_all = None

y_train_all = None
try:
    if os.path.exists(YTRAIN_FILE):
        y_train_all = joblib.load(YTRAIN_FILE)
        if isinstance(y_train_all, (pd.DataFrame, pd.Series)):
            y_train_all = np.asarray(y_train_all).ravel()
        st.sidebar.info(f"y_train: {len(y_train_all)}")
    elif os.path.exists("y_train.npy"):
        y_train_all = np.load("y_train.npy").ravel()
except Exception as e:
    st.sidebar.warning(f"โหลด y_train ไม่สำเร็จ: {e}")
    y_train_all = None

# Align X/y สำหรับ conformal

def _align_Xy_for_conformal(Xt, y):
    if Xt is None or y is None: return None, None
    y = pd.Series(y)
    n = min(len(Xt), len(y))
    if n <= 0: return None, None
    Xt2 = Xt.iloc[:n].copy().reset_index(drop=True)
    y2  = y.iloc[:n].copy().reset_index(drop=True)
    return Xt2, y2

Xt_for_conf, y_for_conf = _align_Xy_for_conformal(X_train_all, y_train_all)

# ==========================
# Confidence bootstrap
# ==========================
NUM_ONLY = ["Area_sqm","Project_Age_notreal","Floors","Total_Units","Launch_Month_sin","Launch_Month_cos"]

conf_ready = False
r_scaler = s_scaler = None
Xt_scaled_train = None
counts_per_feature = None
try:
    if (X_train_all is not None) and (len(X_train_all) > 0):
        num_cols = _resolve_num_cols(X_train_all)
        Xt_num = _prep_num(X_train_all[num_cols], num_cols)
        r_scaler, s_scaler = _robust_scale_fit_nums(Xt_num)
        Xt_scaled_train = _robust_scale_transform_nums(Xt_num, r_scaler, s_scaler)
        # เตรียม categorical counts สำหรับ similarity
        counts_per_feature = _build_counts_per_feature(X_train_all, CAT_FOR_CONF)
        conf_ready = True
except Exception as e:
    st.sidebar.warning(f"ตั้งค่า Confidence ไม่สำเร็จ: {e}")
    conf_ready = False

# ==========================
# Conformal Calibration
# ==========================

def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    n = len(residuals)
    if n <= 0: return float("nan")
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
    Xc = ensure_columns(Xc, ALL_FEATURES, fill_value_map=None)
    yhat_c = np.ravel(pipeline.predict(Xc))
    resid = np.abs(yc - yhat_c)
    return {"q90": _conformal_quantile(resid, 0.10),
            "q95": _conformal_quantile(resid, 0.05),
            "n_calib": int(len(resid))}

conformal_ready, conformal_info = False, None
try:
    if (Xt_for_conf is not None) and (y_for_conf is not None):
        Xt_full = ensure_columns(Xt_for_conf, ALL_FEATURES, fill_value_map=None)
        conformal_info = fit_conformal_from_calib(pipeline, Xt_full, y_for_conf, calib_frac=0.2, seed=42)
        conformal_ready = True
        st.sidebar.success(f"Conformal ✅ calib_n={conformal_info['n_calib']}, q90={conformal_info['q90']:.3f}, q95={conformal_info['q95']:.3f}")
    else:
        st.sidebar.warning("⚠️ Conformal ยังไม่พร้อม (Xt/y ว่างหรือยาวไม่พอ)")
except Exception as e:
    st.sidebar.warning(f"ตั้งค่า Conformal ไม่สำเร็จ: {e}")

# ==========================
# ---------- UI ----------
# ==========================

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

province = flexible_selectbox("จังหวัด - Province", sorted(PROV_TO_DIST))
district = flexible_selectbox("เขต/อำเภอ - District", PROV_TO_DIST.get(province, []))
subdistrict = flexible_selectbox("แขวง/ตำบล - Subdistrict", DIST_TO_SUB.get(district, []))
street = flexible_selectbox("ถนน - Street", SUB_TO_STREET.get(subdistrict, []))

# Zone (auto)
if isinstance(X_train_all, pd.DataFrame) and len(X_train_all) > 0:
    try:
        needed = ["Province","District","Subdistrict","Street","Zone"]
        xtrain_geo = X_train_all.copy()
        for c in needed:
            if c not in xtrain_geo.columns: xtrain_geo[c] = ""
        zone_guess, zone_cands, picked_from = guess_zone(
            province=province, district=district, subdistrict=subdistrict, street=street,
            xtrain_df=xtrain_geo, street_to_zone=STREET_TO_ZONE, topk=6
        )
        options = []
        if zone_guess: options.append(zone_guess)
        for z, _cnt in zone_cands:
            if z and z not in options: options.append(z)
        z_map = STREET_TO_ZONE.get(street, "")
        if z_map and z_map not in options: options.append(z_map)
        options.append("อื่น ๆ (พิมพ์เอง)")
        zone_choice = st.selectbox("Zone (auto, ปรับได้)", options=options, index=0)
        if zone_choice == "อื่น ๆ (พิมพ์เอง)":
            zone = st.text_input("พิมพ์ Zone เอง", value=zone_guess or z_map or "")
        else:
            zone = zone_choice
        with st.expander("ดูเหตุผลการเดา Zone", expanded=False):
            st.write(f"picked_from: **{picked_from or 'fallback'}**")
            if zone_cands:
                st.write("Top zones ใกล้เคียงจากข้อมูลฝึก:")
                st.table(pd.DataFrame(zone_cands, columns=["Zone","Count"]))
    except Exception as e:
        st.warning(f"เดา Zone อัตโนมัติไม่ได้: {e}")
        zone = st.text_input("Zone (manual)", value=STREET_TO_ZONE.get(street, ""))
else:
    zone = st.text_input("Zone (auto from street / editable)", value=STREET_TO_ZONE.get(street, ""))

room_type_base = st.selectbox("ประเภทห้อง — Room_Type", options = [
    'STUDIO','2BED','3BED','1BED','1BED_PLUS','PENTHOUSE','2BED_DUPLEX',
    '1BED_DUPLEX','DUPLEX_OTHER','4BED','POOL_VILLA','4BED_PENTHOUSE',
    '3BED_DUPLEX','1BED_LOFT','3BED_TRIPLEX','3BED_PENTHOUSE','4BED_DUPLEX',
    '5BED_DUPLEX','2BED_PLUS','PENTHOUSE_DUPLEX','Pool Access(เชื่อมสระว่ายน้ำ)',
    '5BED','MOFF-Design','25BED','LOFT_OTHER','2BED_PENTHOUSE','SHOP',
    '1BED_PLUS_LOFT','2BED_LOฟ','Stuio vertiplex','3BED_PLUS','3BED_PLUS_DUPLEX',
    '3BED_LOFT','4BED_LOFT','DUO','1BED_TRIPLEX','1BED_PLUS_TRIPLEX','2BED_TRIPLEX','Simplex'
])

# Row & X
row = {
    "Area_sqm": area, "Project_Age_notreal": age, "Floors": floors, "Total_Units": total_units,
    "Launch_Month_sin": m_sin, "Launch_Month_cos": m_cos,
    "Province": province, "District": district, "Subdistrict": subdistrict, "Street": street,
    "Zone": zone, "Room_Type_Base": room_type_base,
    "is_pool_access": 0, "is_corner": 0, "is_high_ceiling": 0,
}
X = pd.DataFrame([row], columns=ALL_FEATURES)

# Unseen categories (normalized)
unseen_cols = []
if isinstance(X_train_all, pd.DataFrame) and len(X_train_all) > 0:
    for col in ALL_FEATURES:
        if col not in X.columns or col not in X_train_all.columns: continue
        user_value = X[col].iloc[0]
        if X[col].dtype == 'object' or isinstance(user_value, str):
            norm_user = _norm_obj(user_value)
            if norm_user not in _unique_normalized(X_train_all[col]):
                unseen_cols.append(col)
if unseen_cols:
    st.warning(f"⚠️ ค่าใหม่ที่ไม่เคยเจอใน training (หลัง normalize): {', '.join(unseen_cols)}")

# ---------- Predict ----------
if st.button("Predict Price (ล้านบาท)"):
    try:
        # Predict
        y_pred = pipeline.predict(X)
        pred_val = float(np.ravel(y_pred)[0])
        st.metric("ราคาคาดการณ์ (ล้านบาท)", f"{pred_val:.3f}")

        price_per_sqm = (pred_val * 1_000_000.0) / max(1.0, safe_float(area, 1.0))
        st.metric("ราคาต่อตารางเมตร (บาท/ตร.ม.)", f"{price_per_sqm:,.0f}")

        # Router side
        try:
            side_label, side_prob = pipeline.predict_side(X)
            side_txt = side_label[0]
            st.caption(f"Router side: **{side_txt}**  (P[LUX]={side_prob[0]:.2f})")
        except Exception:
            pass

        # Conformal PI (keep same)
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

        # ===== New Hybrid Confidence (Similarity-based) =====
        if conf_ready and (counts_per_feature is not None) and (Xt_scaled_train is not None):
            try:
                # Numeric similarity (RBF on scaled numerics)
                num_cols = _resolve_num_cols(X_train_all)
                x_num = _prep_num(X[num_cols], num_cols)
                x_scaled = _robust_scale_transform_nums(x_num, r_scaler, s_scaler)
                S_num = _rbf_numeric_similarity(x_scaled.values[0], Xt_scaled_train,k=CONF_PARAMS["k_num"], tau=CONF_PARAMS["tau"])


                # Categorical similarity (+ unseen penalties)
                S_cat, unseen_frac = _categorical_similarity(
                    X.iloc[0], counts_per_feature, CAT_FOR_CONF,
                    alpha=CONF_PARAMS["alpha"], m=CONF_PARAMS["m"]
                )

                # Combine + Nonlinear mapping to confidence
                conf_val, S_tilde = _combine_confidence(
                    S_num, S_cat, unseen_frac, CONF_PARAMS
                )

                # Label/report
                conf_rescaled, conf_label, conf_icon = _rescale_and_label(conf_val, low=0.20, high=0.85)
                st.metric("ความมั่นใจของโมเดล (Hybrid Confidence)", f"{conf_rescaled*100:.1f} %", help=f"Label: {conf_label}")
                st.info(f"{conf_icon} ระดับความคล้าย: **{conf_label}**  (S_num={S_num:.3f}, S_cat={S_cat:.3f}, unseen={unseen_frac:.2f})")

                with st.expander("รายละเอียดความคล้าย (กดเพื่อเปิด)", expanded=False):
                    st.write(f"- Numeric similarity (RBF): **{S_num*100:.1f}%**")
                    st.write(f"- Categorical similarity: **{S_cat*100:.1f}%**")
                    st.write(f"- Unseen fraction: **{unseen_frac*100:.1f}%**")
                    st.caption(f"S_tilde = (wN*S_num + wC*S_cat) * exp(-beta * unseen) = {S_tilde:.3f}")
                    st.caption(f"params: wN={CONF_PARAMS['wN']}, wC={CONF_PARAMS['wC']}, alpha={CONF_PARAMS['alpha']}, m={CONF_PARAMS['m']}, beta={CONF_PARAMS['beta']}, lam={CONF_PARAMS['lam']}, gamma={CONF_PARAMS['gamma']}")

                if conf_label == "ต่างเยอะ":
                    with st.expander("🔎 ทำไมความมั่นใจต่ำ? (numeric drift top-3)", expanded=False):
                        dr = _dimension_drift_report(X_train_all, X, NUM_ONLY, topn=3)
                        if dr:
                            st.table(pd.DataFrame(dr, columns=["Column","|z|","Input value"]))
                        if unseen_cols:
                            st.write("หมวดหมู่ที่ไม่เคยพบใน training (หลัง normalize): ", ", ".join(unseen_cols))
            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ Hybrid Confidence ได้: {e}")
        else:
            st.warning("⚠️ ข้อมูล train ไม่พร้อม — ยังไม่แสดง Hybrid Confidence")

    except Exception as e:
        st.error(f"ทำนายไม่สำเร็จ: {e}")

