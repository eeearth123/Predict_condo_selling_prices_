# streamlit_app.py (อัปเดตใหม่)
import os, math, json, sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from locations_th import PROV_TO_DIST, DIST_TO_SUB, SUB_TO_STREET, STREET_TO_ZONE

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
from sklearn.metrics.pairwise import euclidean_distances

def compute_confidence(X_train, X_input, k=5):
    if X_train is None:
        return None
    try:
        distances = euclidean_distances(X_train, X_input)
        topk_distances = np.sort(distances, axis=0)[:k]
        avg_distance = np.mean(topk_distances)
        max_d = np.max(distances)
        score = 1.0 - (avg_distance / max_d)
        return max(0.0, min(1.0, score))  # ค่าระหว่าง 0-1
    except:
        return None


# ---------- โหลดโมเดล ----------
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
# ---------- โหลด X_train (สำหรับใช้คำนวณ Confidence) ----------
try:
    X_train_all = joblib.load("X_train.pkl")  # ต้องมีไฟล์นี้ในโฟลเดอร์เดียวกับ pipeline.pkl
except Exception as e:
    st.warning("⚠️ ไม่พบไฟล์ X_train.pkl — จะไม่แสดง Confidence Score")
    X_train_all = None

# ---------- UI ----------
st.title("🏢 Condo Price Predictor")
st.caption("กรอกข้อมูล → ทำนายราคาขาย (ล้านบาท) และราคาต่อตรางเมตร")

col1, col2, col3 = st.columns(3)
with col1:
    area_input = st.text_input("พื้นที่ (ตร.ม.) — Area_sqm", value="30")
try:
    area = float(area_input)
except:
    area = 0.0  # หรือค่า default อื่น ๆ
floors_input = st.text_input("ชั้นอาคาร — Floors", value="8")
try:
    floors = int(floors_input)
except:
    st.warning("กรุณาใส่จำนวนชั้น เช่น 5")
    floors = 1  # ค่า fallback
    

with col2:
    age_input = st.text_input("อายุโครงการ (ปี) — Project_Age", value="0")
try:
    age = float(age_input)
except:
    st.warning("กรุณาใส่อายุโครงการ เช่น 2.5")
    age = 0.0
total_units_input = st.text_input("จำนวนยูนิตทั้งหมด — Total_Units", value="300")
try:
    total_units = int(total_units_input)
except:
    st.warning("กรุณาใส่จำนวนยูนิต เช่น 200")
    total_units = 100  # fallback ถ้าใส่ไม่ถูก

    

with col3:
    month = st.selectbox("เดือนเปิดตัว - Launch Month", options=list(range(1,13)), index=0)
    
m_sin, m_cos = month_to_sin_cos(month)

# จังหวัด → อำเภอ → ตำบล
province = st.selectbox("จังหวัด - Province", options=sorted(PROV_TO_DIST))
district = st.selectbox("เขต(อำเภอ) - District", options=PROV_TO_DIST.get(province, []))
subdistrict = st.selectbox("แขวง(ตำบล) - Subdistrict", options=DIST_TO_SUB.get(district, []))
street = st.selectbox("ถนน - Street", options=SUB_TO_STREET.get(subdistrict, []))
zone = STREET_TO_ZONE.get(street, "")
st.text_input("Zone (auto)", value=zone, disabled=True)

room_type_base = st.selectbox(
    "ประเภทห้อง - Room_Type", 
    options = [
    'STUDIO', '2BED', '3BED', '1BED', '1BED_PLUS', 'PENTHOUSE', '2BED_DUPLEX',
    '1BED_DUPLEX', 'DUPLEX_OTHER', '4BED', 'POOL_VILLA', '4BED_PENTHOUSE',
    '3BED_DUPLEX', '1BED_LOFT', '3BED_TRIPLEX', '3BED_PENTHOUSE', '4BED_DUPLEX',
    '5BED_DUPLEX', '2BED_PLUS', 'PENTHOUSE_DUPLEX',
    'Pool Access(เชื่อมสระว่ายน้ำ)', '5BED', 'MOFF-Design', '25BED', 'LOFT_OTHER',
    '2BED_PENTHOUSE', 'SHOP', '1BED_PLUS_LOFT', '2BED_LOFT',
    'Stuio  vertiplex', '3BED_PLUS', '3BED_PLUS_DUPLEX', '3BED_LOFT', '4BED_LOFT',
    'DUO', '1BED_TRIPLEX', '1BED_PLUS_TRIPLEX', '2BED_TRIPLEX', 'Simplex'
]

)

# ---------- ย้ายกลุ่ม checkbox มาไว้ล่าง ----------
st.subheader("ลักษณะห้องเพิ่มเติม")
is_pool_access = st.checkbox("ห้องเชื่อมสระว่ายน้ำ (Pool Access)")
is_corner = st.checkbox("ห้องมุม (Corner Room)")
is_high_ceiling = st.checkbox("ห้องเพดานสูง (High Ceiling)")

# ---------- DataFrame ----------
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

X = pd.DataFrame([row], columns=ALL_FEATURES)

with st.expander("ดูข้อมูล (X)"):
    st.dataframe(X, use_container_width=True)

st.divider()

# ---------- Predict ----------
if st.button("Predict Price (ล้านบาท)"):
    st.metric("ราคาคาดการณ์ (ล้านบาท)", f"{pred_val:.3f}")
    price_per_sqm = (pred_val * 1_000_000.0) / max(1.0, safe_float(area, 1.0))
    st.metric("ราคาต่อตารางเมตร (บาท/ตร.ม.)", f"{price_per_sqm:,.0f}")
if X_train_all is not None:
        try:
            X_train_used = X_train_all[ALL_FEATURES].copy()
            confidence = compute_confidence(X_train_used, X[ALL_FEATURES])
if confidence is not None:st.metric("ความมั่นใจของโมเดล (Confidence)", f"{confidence * 100:.1f} %")
except Exception as e:st.warning(f"ไม่สามารถคำนวณ confidence ได้: {e}")


















