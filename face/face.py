import cv2
import numpy as np
import dlib
import streamlit as st
from deepface import DeepFace

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect and draw advanced facial recognition features
def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor
    
    for face in faces:
        landmarks = predictor(gray, face)
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        
        # Draw futuristic grid and points
        for (x, y) in points:
            cv2.circle(overlay, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        
        # Define landmark connections
        connections = [
            (0, 16), (17, 21), (22, 26), (27, 30), (31, 35),
            (36, 41), (42, 47), (48, 59), (60, 67)
        ]
        
        # Draw connected lines
        for start, end in connections:
            for i in range(start, end):
                cv2.line(overlay, points[i], points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        
        # Adding AI-based face recognition info
        try:
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            age = analysis[0]['age']
            gender = analysis[0]['dominant_gender']
            emotion = analysis[0]['dominant_emotion']
            cv2.putText(overlay, f'Age: {age}, Gender: {gender}', (face.left(), face.top() - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f'Emotion: {emotion}', (face.left(), face.top() - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        except:
            pass
    
    # Blend overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Streamlit App UI
def main():
    st.set_page_config(page_title="Advanced Face Recognition", layout="wide")
    st.markdown("""
        <style>
            .css-1d391kg {background-color: #1E1E1E !important; color: white !important;}
            .stButton>button {border-radius: 10px; padding: 10px; font-size: 18px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Advanced Face Recognition System")
    run = st.button("Start Face Detection")
    
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_landmarks(frame)
            stframe.image(frame, channels="BGR")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
