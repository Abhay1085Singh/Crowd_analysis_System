import streamlit as st
import cv2
import tempfile
import os
import time
import base64
from src.detector import CrowdDetector
from src.analytics import CrowdAnalytics


st.set_page_config(page_title="CrowdIntel AI Dashboard", layout="wide")
st.title("🏃‍♂️ AI Crowd Analysis & Safety Alert System")


if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0


st.sidebar.header("🕹️ Control Panel")
conf = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)

st.sidebar.subheader("Crowd Range Limits")
crowd_range = st.sidebar.slider("Define (Low-Mod | Mod-High)", 0, 100, (15, 40))
low_limit, high_limit = crowd_range

enable_audio = st.sidebar.toggle("🔊 Enable Audio Alarm", value=True)


def play_alarm():
    alarm_path = "assets/alert.mp3"
    if os.path.exists(alarm_path):
        with open(alarm_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)


@st.cache_resource
def load_tools(_conf):
    return CrowdDetector(conf=_conf), CrowdAnalytics()

detector, analytics = load_tools(conf)

uploaded_file = st.file_uploader("Upload a video for analysis", type=["mp4", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    analytics.reset_annotators()

    col1, col2 = st.columns([3, 1])
    video_spot = col1.empty()
    metrics_spot = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detections = detector.get_detections(frame)
        count = len(detections)
        
        status, st_color = analytics.get_density_info(count, low_limit, high_limit)
        
        if status == "High" and enable_audio:
            curr_time = time.time()
            if curr_time - st.session_state.last_alarm_time > 5:
                play_alarm()
                st.session_state.last_alarm_time = curr_time
                
        annotated_frame = analytics.annotate_frame(frame, detections)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        video_spot.image(rgb_frame, channels="RGB", use_container_width=True)
        
        with metrics_spot.container():
            st.metric("Live Count", count)
            st.markdown(f"Status: **:{st_color}[{status} Density]**")
            if status == "High":
                st.error("🚨 EMERGENCY: MAX CAPACITY REACHED")

    cap.release()
    st.success("Analysis Complete!")
    os.remove(video_path) 