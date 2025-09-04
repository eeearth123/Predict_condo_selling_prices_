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

NUM_FEATURES = [
    "Area_sqm", "Project_Age_notreal", "Floors", "Total_Units",
    "Launch_Month_sin", "Launch_Month_cos",
    "is_pool_access", "is_corner", "is_high_ceiling"
]
CAT_FEATURES = [
    "Room_Type_Base", "Province", "District", "Subdistrict", "Street", "Zone"
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
PIPELINE_FILE = "pipeline.pkl"

# ---------- Helpers ----------
def month_to_sin_cos(m: int):
    rad = 2 * math.pi * (m - 1) / 12.0
    return math.sin(rad), math.cos(rad)

def safe_float(x, default=0.0):
    try: return float(x)
    except: return float(default)


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def compute_confidence(X_train, X_new, top_k=5):
    try:
        # รวมข้อมูลเพื่อ normalize พร้อมกัน
        combined = pd.concat([X_train, X_new], axis=0)

        # Normalize (เฉพาะตัวเลข)
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)

        X_train_scaled = combined_scaled[:-1]
        X_new_scaled = combined_scaled[-1].reshape(1, -1)

        # คำนวณ cosine similarity
        sim = cosine_similarity(X_train_scaled, X_new_scaled).ravel()  # shape = (n_train,)

        # หาค่าเฉลี่ยของ top-k similarity
        top_k = min(top_k, len(sim))  # กันกรณี train น้อย
        top_scores = np.sort(sim)[-top_k:]
        confidence = float(np.mean(top_scores))

        return confidence

    except Exception as e:
        return None
def flexible_selectbox(label, options):
    """เลือกจาก list หรือพิมพ์เองได้ (return ค่าที่เลือก/พิม)"""
    extended_options = options + ["อื่น ๆ (พิมพ์เอง)"]
    choice = st.selectbox(label, extended_options)
    if choice == "อื่น ๆ (พิมพ์เอง)":
        manual_value = st.text_input(f"กรุณาพิมพ์ {label} ที่ต้องการ")
        return manual_value.strip()
    else:
        return choice


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


# 🌐 Zone (auto from street)
zone = STREET_TO_ZONE.get(street, "")
st.text_input("Zone (auto)", value=zone)







room_type_base = st.selectbox("ประเภทห้อง — Room_Type", options = [
    'STUDIO', '2BED', '3BED', '1BED', '1BED_PLUS', 'PENTHOUSE', '2BED_DUPLEX',
    '1BED_DUPLEX', 'DUPLEX_OTHER', '4BED', 'POOL_VILLA', '4BED_PENTHOUSE',
    '3BED_DUPLEX', '1BED_LOFT', '3BED_TRIPLEX', '3BED_PENTHOUSE', '4BED_DUPLEX',
    '5BED_DUPLEX', '2BED_PLUS', 'PENTHOUSE_DUPLEX', 'Pool Access(เชื่อมสระว่ายน้ำ)',
    '5BED', 'MOFF-Design', '25BED', 'LOFT_OTHER', '2BED_PENTHOUSE', 'SHOP',
    '1BED_PLUS_LOFT', '2BED_LOFT', 'Stuio vertiplex', '3BED_PLUS', '3BED_PLUS_DUPLEX',
    '3BED_LOFT', '4BED_LOFT', 'DUO', '1BED_TRIPLEX', '1BED_PLUS_TRIPLEX',
    '2BED_TRIPLEX', 'Simplex'])

# ---------- Extra Options ----------
st.subheader("ลักษณะห้องเพิ่มเติม")
is_pool_access = st.checkbox("ห้องเชื่อมสระว่ายน้ำ (Pool Access)")
is_corner = st.checkbox("ห้องมุม (Corner Room)")
is_high_ceiling = st.checkbox("ห้องเพดานสูง (High Ceiling)")

# ✅ สร้าง row ก่อน
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
    "is_pool_access": int(is_pool_access),
    "is_corner": int(is_corner),
    "is_high_ceiling": int(is_high_ceiling),
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
        y_pred = pipeline.predict(X)
        pred_val = float(np.ravel(y_pred)[0])
        st.metric("ราคาคาดการณ์ (ล้านบาท)", f"{pred_val:.3f}")

        price_per_sqm = (pred_val * 1_000_000.0) / max(1.0, safe_float(area, 1.0))
        st.metric("ราคาต่อตารางเมตร (บาท/ตร.ม.)", f"{price_per_sqm:,.0f}")

        # ✅ ลองคำนวณ Confidence Score ถ้ามีข้อมูลเทรน
        if X_train_all is not None:
            try:
                # ✅ เตรียม X_train ที่ใช้ encoder เดียวกัน
                X_train_used = X_train_all[ALL_FEATURES].copy()
                if hasattr(pipeline, 'mass_encoder'):
                    X_train_used[CAT_FEATURES] = pipeline.mass_encoder.transform(X_train_used[CAT_FEATURES])
                    X_input = X[ALL_FEATURES].copy()
                    X_input[CAT_FEATURES] = pipeline.mass_encoder.transform(X_input[CAT_FEATURES])
                else:
                    X_input = X[ALL_FEATURES].copy()

                confidence = compute_confidence(X_train_used, X_input)

                if confidence is not None:
                    st.metric("ความมั่นใจของโมเดล (Confidence)", f"{confidence * 100:.1f} %")
                    if confidence >= 0.9:
                        st.success("✅ ข้อมูลคล้ายกับที่โมเดลเคยเห็น → เชื่อมั่นได้สูง")
                    elif confidence >= 0.7:
                        st.info("ℹ️ ข้อมูลใกล้เคียง → น่าเชื่อถือปานกลาง")
                    else:
                        st.warning("⚠️ ข้อมูลแตกต่าง → ระวัง โมเดลอาจไม่แม่น")

            except Exception as e:
                st.warning(f"ไม่สามารถคำนวณ confidence ได้: {e}")

    except Exception as e:
        st.error(f"ทำนายไม่สำเร็จ: {e}")




























