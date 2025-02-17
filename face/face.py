import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
import time
import os
import sqlite3
from datetime import datetime
import plotly.express as px
from gtts import gTTS
import base64

# ---------- إعدادات أساسية ----------
st.set_page_config(
    page_title="Train with Me Pro",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- قاعدة البيانات ----------
conn = sqlite3.connect('fitness.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS workouts
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              exercise TEXT,
              reps INTEGER,
              duration REAL,
              calories REAL,
              date TIMESTAMP)''')
conn.commit()

# ---------- CSS مخصص ----------
st.markdown("""
<style>
    :root {
        --primary: #2A9D8F;
        --secondary: #264653;
        --accent: #E9C46A;
        --background: #FFFFFF;
    }
    
    .main {background-color: var(--background);}
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background: linear-gradient(145deg, var(--primary), var(--secondary));
        color: white !important;
    }
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: var(--accent);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# ---------- وظائف مساعدة ----------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle if angle < 180 else 360-angle

def save_workout(exercise, reps, duration, calories):
    c.execute('''INSERT INTO workouts (exercise, reps, duration, calories, date)
                 VALUES (?, ?, ?, ?, ?)''',
              (exercise, reps, duration, calories, datetime.now()))
    conn.commit()

def get_historical_data():
    return pd.read_sql('SELECT * FROM workouts ORDER BY date DESC', conn)

# ---------- مكونات الواجهة ----------
def user_profile():
    with st.sidebar.expander("👤 الملف الشخصي"):
        weight = st.number_input("الوزن (كجم)", min_value=30, max_value=150, value=70)
        height = st.number_input("الطول (سم)", min_value=140, max_value=220, value=170)
        age = st.number_input("العمر", min_value=10, max_value=100, value=25)
    return weight, height, age

def exercise_instructions(exercise):
    instructions = {
        "Biceps Curl": "• حافظ على استقامة الظهر\n• ثبت الكتفين\n• تحكم في الحركة",
        "Squat": "• حافظ على استقامة الظهر\n• الركبة خلف الأصابع\n• اخفض الجسم حتى 90 درجة"
    }
    st.sidebar.markdown(f"## تعليمات التمرين\n{instructions[exercise]}")

# ---------- الوظائف الرئيسية ----------
def bicep_curls_analysis(landmarks, mp_pose, image, counter, stage, form_corrections):
    # تحليل الزوايا للذراعين
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    # زاوية الكوع
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # زاوية الكتف للتأكد من ثباته
    shoulder_angle = calculate_angle(left_elbow, left_shoulder, 
                                    [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    
    # منطق العد
    if elbow_angle > 160:
        stage = "down"
    if elbow_angle < 30 and stage == "down":
        stage = "up"
        counter += 1
        # تشغيل صوت للعد
        os.system('echo -n "\a"')
    
    # تصحيح الوضعية
    if shoulder_angle > 20:
        form_corrections.append("حرك الكتف!")
        cv2.putText(image, "Shoulder Stability!", (10, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    return counter, stage, form_corrections

def realtime_feedback(col, reps, calories, duration, form_errors):
    with col:
        st.markdown(f'<div class="metric-card"><h3>🔄 التكرارات</h3><h1>{reps}</h1></div>', 
                   unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card"><h3>🔥 السعرات</h3><h1>{calories:.1f}</h1></div>',
                   unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card"><h3>⏱ المدة</h3><h1>{duration}s</h1></div>',
                   unsafe_allow_html=True)
        
        if form_errors:
            st.error("⚠️ تصحيح الوضعية: " + ", ".join(form_errors))

# ---------- الواجهة الرئيسية ----------
def main():
    st.title('🏋️ Train with Me Pro')
    weight, height, age = user_profile()
    
    # إعدادات التمرين
    with st.sidebar:
        exercise_type = st.selectbox("التمرين", ["Biceps Curl", "Squat", "Push-Ups", "Plank"])
        difficulty = st.radio("المستوى", ["مبتدئ", "متوسط", "محترف"])
        
    # واجهة التدريب
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 بدء التمرين", use_container_width=True):
            with st.spinner("جاري إعداد النظام..."):
                start_exercise(col1, col2, exercise_type, weight)

    with col2:
        st.write("## الإرشادات")
        exercise_instructions(exercise_type)
        
    # علامات تبويب الإحصائيات
    with st.expander("📈 التحليلات التاريخية"):
        df = get_historical_data()
        if not df.empty:
            fig = px.line(df, x='date', y='reps', color='exercise', 
                         title="تطور الأداء")
            st.plotly_chart(fig)
        else:
            st.info("لا توجد بيانات سابقة")

def start_exercise(col1, col2, exercise, weight):
    # إعدادات الكاميرا
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    # متغيرات التتبع
    counter = 0
    stage = None
    start_time = time.time()
    form_corrections = []
    
    # عناصر الواجهة
    video_placeholder = col1.empty()
    status_placeholder = col2.empty()
    progress_bar = col2.progress(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # معالجة الصورة
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            # تحليل الحركة
            if exercise == "Biceps Curl":
                counter, stage, form_corrections = bicep_curls_analysis(
                    results.pose_landmarks.landmark,
                    mp_pose,
                    image,
                    counter,
                    stage,
                    form_corrections
                )
            
            # رسم الهيكل العظمي
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        # تحديث الواجهة
        video_placeholder.image(image, channels="RGB")
        
        # حساب المقاييس
        duration = time.time() - start_time
        calories = counter * (weight * 0.05)  # معادلة تقديرية
        
        # تحديث الشريط الجانبي
        realtime_feedback(col2, counter, calories, int(duration), form_corrections)
        progress_bar.progress(min(int(duration % 60)/60, 1.0))
        
        # إيقاف بعد 5 دقائق
        if duration > 300:
            break
    
    # حفظ النتائج
    save_workout(exercise, counter, duration, calories)
    col1.success(f"✅ تم الانتهاء! التكرارات: {counter}")

if __name__ == "__main__":
    main()
