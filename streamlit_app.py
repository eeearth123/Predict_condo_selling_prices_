# =========================
# streamlit_app.py (FULL, Hybrid Confidence v2 + RoomType resolver)
# =========================
import os, sys, math, json, warnings, re
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from locations_th import PROV_TO_DIST, DIST_TO_SUB, SUB_TO_STREET, STREET_TO_ZONE
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, field

# ---------- Setup ----------
st.set_page_config(page_title="Condo Price Predictor", page_icon="🏢", layout="wide")

FLAGS = ["is_pool_access", "is_corner", "is_high_ceiling"]  # จะล็อก = 0 เสมอ
NUM_FEATURES = [
    "Area_sqm", "Project_Age_notreal", "Floors", "Total_Units",
    "Launch_Month_sin", "Launch_Month_cos",
] + FLAGS
CAT_FEATURES = ["Province", "District", "Subdistrict", "Street", "Zone", "Room_Type_Base"]  # default
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

PIPELINE_FILE = "pipeline.pkl"
XTRAIN_FILE   = "X_train.pkl"
YTRAIN_FILE   = "y_train.pkl"

# หมวดหมู่ที่จะใช้คำนวณความมั่นใจแบบความถี่
CAT_FOR_CONF = ["Province","District","Subdistrict","Street","Zone","Room_Type_Base"]

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

TwoSegmentRegressor.__module__ = "main"
sys.modules['main'] = sys.modules[__name__]

# ==================
# ----- Helpers -----
# ==================
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

def _rescale_and_label(conf_raw: float, low=0.20, high=0.85):
    cr = float(conf_raw)
    conf_rescaled = (cr - low) / (high - low)
    conf_rescaled = float(np.clip(conf_rescaled, 0.0, 1.0))
    if conf_rescaled >= 0.75: return conf_rescaled, "มั่นใจสูง", "✅"
    if conf_rescaled >= 0.45: return conf_rescaled, "ปานกลาง", "ℹ️"
    return conf_rescaled, "มั่นใจต่ำ", "⚠️"

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

def month_to_sin_cos(m: int):
    rad = 2 * math.pi (int(m) - 1) / 12.0  # <-- NOTE: will be fixed below
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
    extended_options = list(options) + ["อื่น ๆ (พิมพ์เอง)"]
    choice = st.selectbox(label, extended_options)
    if choice == "อื่น ๆ (พิมพ์เอง)":
        manual_value = st.text_input(f"กรุณาพิมพ์ {label} ที่ต้องการ")
        return manual_value.strip()
    else:
        return choice

# 🔧 fix month_to_sin_cos typo (overwrite)
def month_to_sin_cos(m: int):
    rad = 2 * math.pi * (int(m) - 1) / 12.0
    return math.sin(rad), math.cos(rad)

# ===== Quantile-based numeric confidence =====
def _fit_numeric_quantiles(X_train_all: pd.DataFrame, num_cols: list, qs=(0.05, 0.25, 0.75, 0.95)):
    """คำนวณ quantiles ต่อฟีเจอร์จากชุดฝึก: P5,P25,P75,P95"""
    Q = {}
    Xt = _prep_num(X_train_all[num_cols], num_cols)
    for c in num_cols:
        s = Xt[c].dropna().astype(float)
        if len(s) == 0:
            Q[c] = {"p5": 0, "p25": 0, "p75": 0, "p95": 0}
            continue
        p5, p25, p75, p95 = s.quantile([qs[0], qs[1], qs[2], qs[3]]).values
        Q[c] = {"p5": float(p5), "p25": float(p25), "p75": float(p75), "p95": float(p95)}
    return Q

def _conf_num_quantile(x_row: pd.Series, Q: dict, num_cols: list):
    """ให้ความมั่นใจต่อฟีเจอร์ = 1 ใน [P25,P75], ลดเชิงเส้นไป 0.5 ที่ P5/P95, แล้วรวมด้วย geometric mean"""
    eps = 1e-9
    confs = []
    for c in num_cols:
        v = safe_float(x_row.get(c, 0.0))
        p5, p25, p75, p95 = Q[c]["p5"], Q[c]["p25"], Q[c]["p75"], Q[c]["p95"]
        if p25 == p75:
            confs.append(1.0)
            continue
        if v < p25:
            if v <= p5: conf = 0.5
            else:       conf = 0.5 + 0.5 * (v - p5) / max(p25 - p5, eps)
        elif v > p75:
            if v >= p95: conf = 0.5
            else:        conf = 0.5 + 0.5 * (p95 - v) / max(p95 - p75, eps)
        else:
            conf = 1.0
        confs.append(float(np.clip(conf, 0.0, 1.0)))
    if len(confs) == 0: 
        return 1.0
    return float(np.exp(np.mean(np.log(np.clip(confs, 1e-9, 1.0)))))

# ===== Frequency-based categorical confidence =====
def _fit_categorical_stats(X_train_all: pd.DataFrame, cat_cols: list, alpha: float = 5.0):
    """เก็บ counts/N/K/n_max ต่อคอลัมน์ สำหรับ conf ของหมวดหมู่"""
    stats = []
    Xc = X_train_all.copy()
    for c in cat_cols:
        if c not in Xc.columns:
            Xc[c] = ""
        vc = Xc[c].astype(str).str.strip().fillna("").value_counts(dropna=False)
        counts = vc.to_dict()
        N = int(vc.sum()); K = int(len(vc))
        n_max = int(vc.max()) if K > 0 else 0
        stats.append({"col": c, "counts": counts, "N": N, "K": K, "n_max": n_max, "alpha": float(alpha)})
    return stats

def _conf_cat_from_stats(x_row: pd.Series, cat_stats: list):
    """conf_j = log(1 + n + alpha) / log(1 + n_max + alpha), รวมด้วย geometric mean"""
    confs = []
    for st_ in cat_stats:
        c = st_["col"]
        counts = st_["counts"]; n_max = st_["n_max"]; a = st_["alpha"]
        val = str(x_row.get(c, "")).strip()
        n = int(counts.get(val, 0))
        num = math.log(1.0 + n + a)
        den = math.log(1.0 + (n_max if n_max > 0 else 0) + a)
        conf_j = (num / den) if den > 0 else 0.5
        confs.append(float(np.clip(conf_j, 0.0, 1.0)))
    if len(confs) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(np.clip(confs, 1e-9, 1.0)))))

def _hybrid_from_num_cat(conf_num: float, conf_cat: float):
    return float(np.sqrt(max(1e-9, conf_num) * max(1e-9, conf_cat)))

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
# RoomType resolver (NEW)
# ==========================
def _roomtype_col(df):
    if isinstance(df, pd.DataFrame):
        if "Room_Type_Base" in df.columns: return "Room_Type_Base"
        if "RoomType_Group" in df.columns: return "RoomType_Group"
    return "RoomType_Group"  # ดีฟอลต์ถ้าไม่รู้ ให้ไปตาม group ใหม่นี้

ROOMTYPE_COL = _roomtype_col(X_train_all)

def _roomtype_vocab(df, colname):
    if not isinstance(df, pd.DataFrame) or (colname not in df.columns):
        return set()
    return set(df[colname].dropna().astype(str).map(_norm_obj).unique())

ROOMTYPE_VOCAB = _roomtype_vocab(X_train_all, ROOMTYPE_COL)

ROOMTYPE_ALIAS = {
    # UI -> canonical ที่คาดว่าตรงกับเทรน (รองรับเคสตัวพิมพ์เล็ก)
    "1BEDplus": "1BEDplus", "1bedplus": "1BEDplus",
    "1BEDduplex": "1BEDduplex", "1bedduplex": "1BEDduplex",
    "2BEDplus": "2BEDplus", "2bedplus": "2BEDplus",
    "2BEDduplex": "2BEDduplex", "2bedduplex": "2BEDduplex",
    "3BEDplus": "3BEDplus", "3bedplus": "3BEDplus",
    "3BEDduplex": "3BEDduplex", "3bedduplex": "3BEDduplex",
    # ถ้าเทรนใช้รูปแบบอื่น ให้เพิ่มแมปได้ที่นี่
}

# ==========================
# Quantile & Frequency stats (NEW)
# ==========================
quantiles_per_num = None
cat_stats_for_conf = None
conf_ready = False

try:
    if (X_train_all is not None) and (len(X_train_all) > 0):
        # เติมคอลัมน์หมวดหมู่ที่ขาดด้วยค่าว่าง
        for c in CAT_FOR_CONF:
            if c not in X_train_all.columns:
                X_train_all[c] = ""
        # ให้แน่ใจว่ามีคอลัมน์ roomtype ที่ใช้จริงใน CAT_FOR_CONF/ALL_FEATURES
        if ROOMTYPE_COL not in CAT_FOR_CONF:
            CAT_FOR_CONF = [z for z in CAT_FOR_CONF if z not in ("Room_Type_Base","RoomType_Group")] + [ROOMTYPE_COL]
        # เตรียมสถิติ
        num_cols = _resolve_num_cols(X_train_all)
        quantiles_per_num = _fit_numeric_quantiles(X_train_all, num_cols)
        cat_stats_for_conf = _fit_categorical_stats(X_train_all, CAT_FOR_CONF, alpha=5.0)
        conf_ready = True
        st.sidebar.success("Confidence stats (quantiles & categorical frequency) พร้อมใช้งาน ✅")
    else:
        st.sidebar.warning("⚠️ ไม่มี X_train — ยังทำ confidence แบบใหม่ไม่ได้")
except Exception as e:
    st.sidebar.warning(f"ตั้งค่า Quantile/Freq ไม่สำเร็จ: {e}")
    conf_ready = False

with st.sidebar.expander("Confidence Stats Debug", expanded=False):
    st.write("conf_ready:", conf_ready)
    if X_train_all is None:
        st.write("X_train_all: None")
    else:
        st.write("X_train_all shape:", getattr(X_train_all, "shape", None))
        st.write("ROOMTYPE_COL:", ROOMTYPE_COL)
        st.write("Has quantiles_per_num:", quantiles_per_num is not None)
        st.write("Has cat_stats_for_conf:", cat_stats_for_conf is not None)
        st.write("CAT_FOR_CONF:", CAT_FOR_CONF)

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
    Xt_for_conf, y_for_conf = _align_Xy_for_conformal(X_train_all, y_train_all)
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

# ----------------------
# RoomType UI (ใช้ตามที่มีจริง) + resolver → match vocab เทรน
# ----------------------
roomtype_options = [
    "STUDIO","1BED","1BEDplus","1BEDduplex",
    "2BED","2BEDplus","2BEDduplex",
    "3BED","3BEDplus","3BEDduplex",
    "other"
]
rt_user = st.selectbox("ประเภทห้อง — Room_Type", options=roomtype_options)
rt_canon = ROOMTYPE_ALIAS.get(rt_user, rt_user)  # map เป็นชื่อที่โมเดลคาด
rt_norm  = _norm_obj(rt_canon)

# ถ้าไม่อยู่ใน vocab เทรน → fallback เป็น 'other' หรือ top-1
if ROOMTYPE_VOCAB and (rt_norm not in ROOMTYPE_VOCAB):
    st.warning("ปรับ Room Type เป็น 'other' เนื่องจากรูปแบบนี้ไม่พบใน training")
    if _norm_obj("other") in ROOMTYPE_VOCAB:
        rt_canon = "other"
    else:
        top1 = (
            X_train_all[ROOMTYPE_COL]
            .astype(str).str.strip()
            .value_counts(dropna=False)
            .index[0]
        )
        rt_canon = str(top1)

# ---------- Build row ----------
row = {
    "Area_sqm": area, "Project_Age_notreal": age, "Floors": floors, "Total_Units": total_units,
    "Launch_Month_sin": m_sin, "Launch_Month_cos": m_cos,
    "Province": province, "District": district, "Subdistrict": subdistrict, "Street": street,
    "Zone": zone,
    "is_pool_access": 0, "is_corner": 0, "is_high_ceiling": 0,
}
# ใส่ roomtype ตามคอลัมน์เทรนจริง
row[ROOMTYPE_COL] = rt_canon

# เพื่อความเข้ากันได้ ถ้าอีกชื่อยังถูกใช้อยู่ที่อื่น ให้ใส่ทั้งสองคีย์
if ROOMTYPE_COL == "RoomType_Group":
    row["Room_Type_Base"] = rt_canon
elif ROOMTYPE_COL == "Room_Type_Base":
    row["RoomType_Group"] = rt_canon

# อัพเดต FEATURE LIST ให้สอดคล้องคอลัมน์จริง
CAT_FEATURES = [c for c in ["Province","District","Subdistrict","Street","Zone","Room_Type_Base","RoomType_Group"]
                if (c in (X_train_all.columns if isinstance(X_train_all, pd.DataFrame) else []) ) or (c in row)]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

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

        # Router side (optional)
        try:
            side_label, side_prob = pipeline.predict_side(X)
            side_txt = side_label[0]
            st.caption(f"Router side: **{side_txt}**  (P[LUX]={side_prob[0]:.2f})")
        except Exception:
            pass

        # Conformal PI
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

        # ===== Hybrid Confidence (Quantile numeric + Frequency categorical + Geometric mean) =====
        if conf_ready and (quantiles_per_num is not None) and (cat_stats_for_conf is not None):
            try:
                num_cols = _resolve_num_cols(X_train_all)
                conf_num = _conf_num_quantile(X.iloc[0], quantiles_per_num, num_cols)
                conf_cat = _conf_cat_from_stats(X.iloc[0], cat_stats_for_conf)
                hybrid_raw = _hybrid_from_num_cat(conf_num, conf_cat)

                conf_rescaled, conf_label, conf_icon = _rescale_and_label(hybrid_raw, low=0.20, high=0.85)
                st.metric("ความมั่นใจของโมเดล (Hybrid Confidence)", f"{conf_rescaled*100:.1f} %", help=f"Label: {conf_label}")
                st.info(f"{conf_icon} รายละเอียด: conf_num={conf_num:.3f}, conf_cat={conf_cat:.3f} → geometric mean={hybrid_raw:.3f}")

                if conf_label == "มั่นใจต่ำ":
                    with st.expander("🔎 ทำไมความมั่นใจต่ำ?", expanded=False):
                        dr = _dimension_drift_report(X_train_all, X, NUM_ONLY_FALLBACK, topn=3)
                        if dr:
                            st.table(pd.DataFrame(dr, columns=["Column","|z|","Input value"]))
                        if unseen_cols:
                            st.write("หมวดหมู่ที่ไม่เคยพบใน training (หลัง normalize): ", ", ".join(unseen_cols))
            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ Hybrid Confidence (ใหม่) ได้: {e}")
        else:
            st.warning("⚠️ ข้อมูลสถิติไม่พร้อม — ยังไม่แสดง Hybrid Confidence (ใหม่)")

    except Exception as e:
        st.error(f"ทำนายไม่สำเร็จ: {e}")
