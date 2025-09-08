# streamlit_app.py (อัปเดตใหม่ พร้อมภาษาไทย+อังกฤษ)
import os, math, json, sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from locations_th import PROV_TO_DIST, DIST_TO_SUB, SUB_TO_STREET, STREET_TO_ZONE
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


# ---------- Helpers ----------
import re

def _norm_txt(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _top_counts(series, topk=5):
    vc = series.dropna().astype(str).str.strip().value_counts()
    return [(z, int(c)) for z, c in vc.head(topk).items()]

def _filter_chain(df, province=None, district=None, subdistrict=None, street=None):
    """คืน list ของ (label, df_filtered) ตามลำดับความจำเพาะ → กว้าง"""
    steps = []
    # ใช้ .str.lower().str.strip() เทียบแบบ normalize
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
    best_zone = ""
    candidates = []
    picked_from = ""

    # 1) ลองโหวตจากข้อมูลฝึก (ตาม chain: แคบ → กว้าง)
    if xtrain_df is not None and "Zone" in xtrain_df.columns:
        for tag, dff in _filter_chain(xtrain_df, province, district, subdistrict, street):
            if len(dff):
                cands = _top_counts(dff["Zone"], topk=topk)
                if cands:
                    best_zone = cands[0][0]
                    candidates = cands
                    picked_from = tag
                    break

    # 2) fallback: mapping จากถนน
    if not best_zone and street_to_zone is not None:
        z = street_to_zone.get(street, "")
        if z:
            best_zone = z
            candidates = [(z, 0)]
            picked_from = "street_mapping"

    return best_zone, candidates, picked_from

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
    # ถ้ามีคอลัมน์เกินมา ปล่อยไว้ได้ แต่เราจะเลือกเฉพาะที่ต้องใช้
    return df[cols]


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def flexible_selectbox(label, options):
    """เลือกจาก list หรือพิมพ์เองได้ (return ค่าที่เลือก/พิม)"""
    extended_options = options + ["อื่น ๆ (พิมพ์เอง)"]
    choice = st.selectbox(label, extended_options)
    if choice == "อื่น ๆ (พิมพ์เอง)":
        manual_value = st.text_input(f"กรุณาพิมพ์ {label} ที่ต้องการ")
        return manual_value.strip()
    else:
        return choice
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def _which_side(pipeline, X_one_row):
    """บอกว่าพยากรณ์แถวนี้ใช้ MASS หรือ LUX"""
    # ถ้าเป็น TwoSegmentRegressor (area-rule 250 ตร.ม.)
    if hasattr(pipeline, "predict"):
        try:
            # ถ้าคลาสคุณมีเมธอดภายในบอกฝั่งชัด ๆ ก็ใช้เลย
            pass
        except:
            pass
    # fallback แบบง่าย (ถ้ามี Area_sqm)
    return "LUX" if float(X_one_row.get("Area_sqm", 0)) > 250 else "MASS"

def _encode_like_model(pipeline, X_df, cat_cols, side):
    """พยายามเข้ารหัสแบบเดียวกับโมเดล; ถ้าไม่มี ให้ one-hot ชั่วคราว"""
    X_df = X_df.copy()
    if side == "LUX" and hasattr(pipeline, "lux_encoder") and pipeline.lux_encoder is not None:
        X_df[cat_cols] = pipeline.lux_encoder.transform(X_df[cat_cols])
        return X_df, "model"
    if side == "MASS" and hasattr(pipeline, "mass_encoder") and pipeline.mass_encoder is not None:
        X_df[cat_cols] = pipeline.mass_encoder.transform(X_df[cat_cols])
        return X_df, "model"

    # ---- fallback: one-hot ทั้งชุดสำหรับคำนวณความมั่นใจเท่านั้น ----
    pre = ColumnTransformer([
        ("num", "passthrough", [c for c in X_df.columns if c not in cat_cols]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    enc = Pipeline([("pre", pre)])
    X_arr = enc.fit_transform(X_df)   # fit ด้วย train รวม input 1 แถว
    X_encoded = pd.DataFrame(X_arr)
    return X_encoded, "onehot"

def compute_confidence_robust(pipeline, X_train_all, X_input_one, all_cols, cat_cols, top_k=5):
    # เตรียมชุดให้มีคอลัมน์ครบ
    X_train_used = X_train_all.reindex(columns=all_cols).copy()
    X_input = X_input_one.reindex(columns=all_cols).copy()

    # เลือกฝั่งโมเดล (MASS/LUX) สำหรับแถวนี้
    side = _which_side(pipeline, X_input.iloc[0].to_dict())

    # เข้ารหัสให้เป็นตัวเลขแบบเดียวกับโมเดล (หรือ one-hot fallback)
    X_train_enc, mode1 = _encode_like_model(pipeline, X_train_used, cat_cols, side)
    # สำคัญ: ต้องใช้ encoder เดียวกันกับ train เพื่อให้คอลัมน์ตรงกัน
    if mode1 == "model":
        X_input_enc, _ = _encode_like_model(pipeline, X_input, cat_cols, side)
    else:
        # one-hot ใหม่ ต้อง fit จาก train+input พร้อมกันให้คอลัมน์ตรง
        combo = pd.concat([X_train_used, X_input], axis=0)
        X_combo_enc, _ = _encode_like_model(pipeline, combo, cat_cols, side)
        X_train_enc = X_combo_enc.iloc[:-1, :].reset_index(drop=True)
        X_input_enc = X_combo_enc.iloc[-1:, :].reset_index(drop=True)

    # สเกลแล้วคำนวณ cosine
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_new_scaled = scaler.transform(X_input_enc)

    sim = cosine_similarity(X_train_scaled, X_new_scaled).ravel()
    k = min(top_k, len(sim))
    conf = float(np.mean(np.sort(sim)[-k:]))
    return conf


# ---------- Load model ----------
try:
    import two_segment
    sys.modules['main'] = two_segment
except Exception:
    pass

if not os.path.exists(PIPELINE_FILE):
    st.error(f"ไม่พบไฟล์ {PIPELINE_FILE} — กรุณาวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับสคริปต์")
    st.stop()

try:
    pipeline = joblib.load(PIPELINE_FILE)
    st.sidebar.success("โหลด pipeline.pkl สำเร็จ ✅")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ---------- โหลด X_train (สำหรับ Confidence) ----------
try:
    X_train_all = joblib.load("X_train.pkl")
except:
    st.warning("⚠️ ไม่พบ X_train.pkl — จะไม่สามารถแสดง Confidence Score ได้")
    X_train_all = None
# ใส่ไว้ตอนโหลดแอป (หลังโหลด X_train_all)
NUM_ONLY = ["Area_sqm","Project_Age_notreal","Floors","Total_Units","Launch_Month_sin","Launch_Month_cos"]

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np, pandas as pd

# 1) เตรียมสเกลจาก train-only (เฉพาะคอลัมน์ตัวเลข)
def _fit_numeric_scaler(X_train_all, num_cols=NUM_ONLY):
    Xt = X_train_all[num_cols].copy().astype(float)
    scaler = StandardScaler().fit(Xt)
    Xt_scaled = scaler.transform(Xt)
    return scaler, Xt_scaled  # เก็บ Xt_scaled ไว้ใช้คำนวณ distribution

# 2) สร้าง distribution ของ "ความคล้ายภายใน train" (top-k เฉลี่ย)
def _train_similarity_distribution(Xt_scaled, top_k=10):
    # cosine กับตัวเอง แล้วตัดค่าทะแยงออก (ไม่เทียบตัวเอง)
    sim = cosine_similarity(Xt_scaled)
    np.fill_diagonal(sim, -np.inf)
    # เอาค่าเฉลี่ยของ top-k เพื่อนบ้านที่ใกล้สุดสำหรับแต่ละแถว
    topk_mean = np.mean(np.sort(sim, axis=1)[:, -top_k:], axis=1)
    # map เป็นช่วง [0,1] (cosine เป็น [-1,1])
    topk_mean_01 = (topk_mean + 1.0) / 2.0
    return topk_mean_01  # ใช้เป็นฐานเปอร์เซ็นไทล์

# 3) คำนวณ confidence สำหรับอินพุตใหม่ → คืนค่าเปอร์เซ็นไทล์ (0..1)
def confidence_numeric_percentile(X_input, scaler, Xt_scaled_train, dist_ref_01, num_cols=NUM_ONLY, top_k=10):
    x = X_input[num_cols].copy().astype(float)
    x_scaled = scaler.transform(x)
    sim = cosine_similarity(Xt_scaled_train, x_scaled).ravel()
    topk_mean = np.mean(np.sort(sim)[-top_k:])
    conf_01 = (topk_mean + 1.0) / 2.0
    # เปอร์เซ็นไทล์เทียบกับ distribution ของ train
    pct = float((dist_ref_01 <= conf_01).mean())
    return pct  # ใช้เป็น confidence
# สร้างครั้งเดียวตอนเริ่มแอป (หลังโหลด X_train_all)
scaler_num, Xt_scaled_train = _fit_numeric_scaler(X_train_all, NUM_ONLY)
dist_ref_01 = _train_similarity_distribution(Xt_scaled_train, top_k=10)

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

# ✅ ค่อยมาเช็ค unseen values
unseen_cols = []
if 'X_train_all' in globals() and X_train_all is not None:
    for col in ALL_FEATURES:
        if col not in X.columns or col not in X_train_all.columns:
            continue
        user_value = X[col].values[0]
        unique_values = X_train_all[col].unique()

        if X[col].dtype == 'object' and user_value not in unique_values:
            unseen_cols.append(col)

if unseen_cols:
    st.warning(f"⚠️ ค่าต่อไปนี้ไม่เคยปรากฏในการฝึกโมเดล: {', '.join(unseen_cols)}")


# ---------- Predict ----------
if st.button("Predict Price (ล้านบาท)"):
    try:
        # ===== ทำนายราคา =====
        y_pred = pipeline.predict(X)
        pred_val = float(np.ravel(y_pred)[0])
        st.metric("ราคาคาดการณ์ (ล้านบาท)", f"{pred_val:.3f}")

        price_per_sqm = (pred_val * 1_000_000.0) / max(1.0, safe_float(area, 1.0))
        st.metric("ราคาต่อตารางเมตร (บาท/ตร.ม.)", f"{price_per_sqm:,.0f}")

        # ===== คำนวณ Confidence =====
        if X_train_all is not None:
            try:
                # 1) บังคับคอลัมน์ให้ตรงกับที่โมเดลเคยเทรน และเติม FLAGS=0
                fill0 = {"is_pool_access":0, "is_corner":0, "is_high_ceiling":0}
                X_train_used = ensure_columns(X_train_all, ALL_FEATURES, fill_value_map=fill0)
                X_input_used = ensure_columns(X,            ALL_FEATURES, fill_value_map=fill0)

                # 2) เลือกฝั่งโมเดล (ตาม area-rule 250 ตร.ม.)
                side = "LUX" if float(X_input_used.loc[0, "Area_sqm"]) > 250 else "MASS"

                # 3) เข้ารหัสให้เหมือน encoder ที่ใช้จริง
                def encode_with_model(pipeline, Xdf, cat_cols, side):
                    Xdf = Xdf.copy()
                    if side == "LUX" and hasattr(pipeline, "lux_encoder") and pipeline.lux_encoder is not None:
                        Xdf[cat_cols] = pipeline.lux_encoder.transform(Xdf[cat_cols])
                        return Xdf
                    if side == "MASS" and hasattr(pipeline, "mass_encoder") and pipeline.mass_encoder is not None:
                        Xdf[cat_cols] = pipeline.mass_encoder.transform(Xdf[cat_cols])
                        return Xdf
                    return Xdf  # ถ้าไม่มี encoder

                X_train_enc = encode_with_model(pipeline, X_train_used, CAT_FEATURES, side)
                X_input_enc = encode_with_model(pipeline, X_input_used, CAT_FEATURES, side)

                # 4) ถ้ายังเป็น object อยู่ → one-hot ชั่วคราว
                needs_onehot = any(getattr(X_train_enc[c], "dtype", None) == "object" for c in CAT_FEATURES)
                if needs_onehot:
                    from sklearn.preprocessing import OneHotEncoder
                    from sklearn.compose import ColumnTransformer
                    from sklearn.pipeline import Pipeline

                    pre = ColumnTransformer([
                        ("num", "passthrough", [c for c in ALL_FEATURES if c not in CAT_FEATURES]),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
                    ])
                    oh = Pipeline([("pre", pre)])
                    combo = pd.concat([X_train_used, X_input_used], axis=0)
                    combo_arr = oh.fit_transform(combo)
                    combo_df  = pd.DataFrame(combo_arr)

                    X_train_enc = combo_df.iloc[:-1, :].reset_index(drop=True)
                    X_input_enc = combo_df.iloc[-1:, :].reset_index(drop=True)

                # 5) สเกลแล้วคำนวณ cosine similarity
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import cosine_similarity

                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(X_train_enc)
                input_scaled = scaler.transform(X_input_enc)

                sim = cosine_similarity(train_scaled, input_scaled).ravel()
                k = min(5, len(sim))
                confidence = float(np.mean(np.sort(sim)[-k:]))

                conf = confidence_numeric_percentile(X, scaler_num, Xt_scaled_train, dist_ref_01, NUM_ONLY, top_k=10)
                st.metric("ความมั่นใจของโมเดล (Confidence)", f"{conf*100:.1f} %")
                if conf >= 0.9:
                    st.success("✅ คล้ายข้อมูลฝึกมาก")
                elif conf >= 0.7:
                    st.info("ℹ️ ใกล้เคียงพอสมควร")
                else:
                    st.warning("⚠️ ค่อนข้างต่างจากข้อมูลฝึก")

            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ confidence ได้: {e}")
        else:
            st.warning("⚠️ ไม่พบ X_train.pkl — จะไม่สามารถแสดง Confidence Score ได้")

    except Exception as e:
        st.error(f"ทำนายไม่สำเร็จ: {e}")




































