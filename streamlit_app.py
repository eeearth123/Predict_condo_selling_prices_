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
        if p25 == p75:        # กัน corner case
            confs.append(1.0)
            continue
        if v < p25:
            if v <= p5: conf = 0.5
            else:
                conf = 0.5 + 0.5 * (v - p5) / max(p25 - p5, eps)
        elif v > p75:
            if v >= p95: conf = 0.5
            else:
                conf = 0.5 + 0.5 * (p95 - v) / max(p95 - p75, eps)
        else:
            conf = 1.0
        confs.append(float(np.clip(conf, 0.0, 1.0)))
    # geometric mean (ถ้าไม่มีฟีเจอร์ ให้ 1.0)
    if len(confs) == 0: 
        return 1.0
    return float(np.exp(np.mean(np.log(np.clip(confs, 1e-9, 1.0)))))

# ===== Frequency-based categorical confidence (ต่อคอลัมน์) =====

def _fit_categorical_stats(X_train_all: pd.DataFrame, cat_cols: list, alpha: float = 5.0):
    """
    เก็บ stats ต่อคอลัมน์: counts, N, K, n_max สำหรับทำความมั่นใจแบบความถี่
    """
    stats = []
    for c in cat_cols:
        if c not in X_train_all.columns:
            vc = pd.Series(dtype=int)
        else:
            vc = X_train_all[c].astype(str).str.strip().fillna("").value_counts(dropna=False)
        counts = vc.to_dict()
        N = int(vc.sum())
        K = int(len(vc))
        n_max = int(vc.max()) if K > 0 else 0
        stats.append({"col": c, "counts": counts, "N": N, "K": K, "n_max": n_max, "alpha": float(alpha)})
    return stats

def _conf_cat_from_stats(x_row: pd.Series, cat_stats: list):
    """
    ความมั่นใจหมวดหมู่ต่อคอลัมน์: ใช้ log-normalize บน (n + alpha)
    conf_j = log(1 + n + alpha) / log(1 + n_max + alpha)
    รวมทุกคอลัมน์ด้วย geometric mean
    """
    confs = []
    for st in cat_stats:
        c = st["col"]
        counts = st["counts"]; n_max = st["n_max"]; a = st["alpha"]
        val = str(x_row.get(c, "")).strip()
        n = int(counts.get(val, 0))
        num = math.log(1.0 + n + a)
        den = math.log(1.0 + (n_max if n_max > 0 else 0) + a)
        conf_j = (num / den) if den > 0 else 0.5
        confs.append(float(np.clip(conf_j, 0.0, 1.0)))
    if len(confs) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(np.clip(confs, 1e-9, 1.0)))))

# ===== Overall Hybrid (geometric mean ของ numeric & categorical) =====

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
# Quantile & Frequency stats (NEW)
# ==========================
quantiles_per_num = None
cat_stats_for_conf = None

try:
    if (X_train_all is not None) and (len(X_train_all) > 0):
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

        # ===== Hybrid Confidence (NEW: Quantile numeric + Frequency categorical + Geometric mean) =====#
        if conf_ready and (quantiles_per_num is not None) and (cat_stats_for_conf is not None):
            try:
                num_cols = _resolve_num_cols(X_train_all)
                # Numeric confidence (quantile-based)
                conf_num = _conf_num_quantile(X.iloc[0], quantiles_per_num, num_cols)

                # Categorical confidence (frequency-based)
                conf_cat = _conf_cat_from_stats(X.iloc[0], cat_stats_for_conf)

                # Combine with geometric mean
                hybrid_raw = _hybrid_from_num_cat(conf_num, conf_cat)

                # (ออปชัน) ปรับสเกลให้อ่านง่ายขึ้นเป็น % และติด label
                conf_rescaled, conf_label, conf_icon = _rescale_and_label(hybrid_raw, low=0.20, high=0.85)

                st.metric("ความมั่นใจของโมเดล (Hybrid Confidence)", f"{conf_rescaled*100:.1f} %", help=f"Label: {conf_label}")
                st.info(f"{conf_icon} รายละเอียด: conf_num={conf_num:.3f}, conf_cat={conf_cat:.3f} → geometric mean={hybrid_raw:.3f}")

                with st.expander("รายละเอียดการคำนวณ", expanded=False):
                    st.write("- **Numeric (quantile-based)**: 1 ในช่วง [P25,P75], ลดเชิงเส้นไป 0.5 ที่ P5/P95, รวมทุกคอลัมน์ด้วย geometric mean")
                    st.write("- **Categorical (freq-based)**: conf_j=log(1+n+α)/log(1+n_max+α), รวมข้ามคอลัมน์ด้วย geometric mean; α=5")
                    st.caption("Hybrid = sqrt(conf_num * conf_cat)  → จากนั้นทำ rescale (low=0.20, high=0.85) เพื่อแปลงเป็นสเกลที่อ่านง่ายใน UI")
                
                if conf_label == "มั่นใจต่ำ":
                    with st.expander("🔎 ทำไมความมั่นใจต่ำ?", expanded=False):
                        dr = _dimension_drift_report(X_train_all, X, NUM_ONLY, topn=3)
                        if dr:
                            st.table(pd.DataFrame(dr, columns=["Column","|z|","Input value"]))
                        if unseen_cols:
                            st.write("หมวดหมู่ที่ไม่เคยพบใน training (หลัง normalize): ", ", ".join(unseen_cols))
            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ Hybrid Confidence (ใหม่) ได้: {e}")
        else:
            st.warning("⚠️ ข้อมูลสถิติไม่พร้อม — ยังไม่แสดง Hybrid Confidence (ใหม่)")



