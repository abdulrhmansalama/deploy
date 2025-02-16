import os
import bz2
import urllib.request
import cv2
import numpy as np
import dlib
import streamlit as st

# مسار جديد للملف داخل /tmp/ ليكون متاحًا في بيئة Streamlit
MODEL_PATH = "/tmp/shape_predictor_68_face_landmarks.dat"

# Ensure the shape predictor file is downloaded
def download_shape_predictor():
    compressed_path = MODEL_PATH + ".bz2"
    
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading shape predictor model...")
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, compressed_path)
        
        with bz2.BZ2File(compressed_path, "rb") as compressed_file, open(MODEL_PATH, "wb") as output_file:
            output_file.write(compressed_file.read())
        
        st.success("Download complete!")

download_shape_predictor()

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Function to detect and draw facial landmarks
def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor
    
    for face in faces:
        landmarks = predictor(gray, face)
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        
        # Draw circles for each landmark
        for (x, y) in points:
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Define landmark connections
        connections = [
            (0, 16), (17, 21), (22, 26), (27, 30), (31, 35),
            (36, 41), (42, 47), (48, 59), (60, 67)
        ]
        
        # Draw connected lines with anti-aliasing effect
        for start, end in connections:
            for i in range(start, end):
                cv2.line(overlay, points[i], points[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        
        # Close shapes (eyes, lips)
        cv2.line(overlay, points[36], points[41], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, points[42], points[47], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, points[48], points[59], (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, points[60], points[67], (0, 255, 255), 2, cv2.LINE_AA)
    
    # Blend overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Streamlit App
def main():
    st.set_page_config(page_title="Professional Face Recognition", layout="wide")
    st.title("Face Landmark Detection - Professional Look")
    run = st.checkbox("Run Face Detection")
    
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
