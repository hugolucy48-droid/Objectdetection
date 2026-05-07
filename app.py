import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,  # smoother performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
import time
import os

# -----------------------
# EXTRA FEATURES SETUP
# -----------------------
ALERT_OBJECT = "person"
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = 0

if "object_count" not in st.session_state:
    st.session_state.object_count = {}

# -----------------------
# UPDATED CALLBACK (WRAP YOUR EXISTING LOGIC)
# -----------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    current_count = {}

    # -----------------------
    # OBJECT COUNTING + ALERT
    # -----------------------
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Count objects
            current_count[label] = current_count.get(label, 0) + 1

            # ALERT
            if label == ALERT_OBJECT:
                cv2.putText(
                    annotated_frame,
                    f"ALERT: {ALERT_OBJECT.upper()}!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # Save latest counts
    st.session_state.object_count = current_count

    # -----------------------
    # SAVE FRAME
    # -----------------------
    current_time = time.time()
    if current_time - st.session_state.last_save_time > 3:
        filename = f"{SAVE_DIR}/frame_{int(current_time)}.jpg"
        cv2.imwrite(filename, annotated_frame)
        st.session_state.last_save_time = current_time

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# -----------------------
# DISPLAY RESULTS
# -----------------------
st.subheader("📊 Object Count")
st.write(st.session_state.object_count)

st.subheader("🚨 Alert Object")
st.write(ALERT_OBJECT)

st.subheader("💾 Saved Frames")
st.write(f"Saved in folder: {SAVE_DIR}/")