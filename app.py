import streamlit as st
import cv2
import tempfile
import numpy as np
import torch
from torchvision import models, transforms
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import threading


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Based Crowd Density Montior", layout="wide")

# ==============================
# 🎨 CLEAN DASHBOARD CSS
# ==============================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.header {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    color: Red;
}

.card {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #30363D;
}

.metric-title {
    font-size: 16px;
    color: #8B949E;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("<div class='header'>🚀AI Based Crowd Density Monitoring System </div>", unsafe_allow_html=True)

# ==============================
# MODEL LOAD
# ==============================
@st.cache_resource
def load_model():
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
])

PERSON_CLASS = 15

# ==============================
# FUNCTIONS
# ==============================
def process_frame(frame):
    input_tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    pred = output.argmax(0).byte().cpu().numpy()
    mask = (pred == PERSON_CLASS).astype(np.uint8)

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    density = np.sum(mask) / mask.size

    return density, mask


def generate_heatmap(mask):
    heat = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 0)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)


def classify(d):
    if d < 0.03:
        return "LOW", "#2ECC71"
    elif d < 0.05:
        return "MEDIUM", "#F1C40F"
    else:
        return "HIGH", "#E74C3C"

class VideoProcessor(VideoProcessorBase):
    density = 0
    mask = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        density, mask = process_frame(img)
        self.density = density
        self.mask = mask
        overlay = generate_heatmap(mask)
        overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        return av.VideoFrame.from_ndarray(overlay, format="bgr24")

# ==============================
# INPUT
# ==============================
mode = st.radio("Choose mode:", ("Upload Video", "Live Webcam"))

if mode == "Live Webcam":
    st.session_state.densities = []
    densities = []
    st.rerun()

# Define layout first
video_col, heatmap_col = st.columns(2)
metric1, metric2, metric3 = st.columns(3)
chart_container = st.container()

# Placeholders
video_placeholder = video_col.empty()
heatmap_placeholder = heatmap_col.empty()
density_placeholder = metric1.empty()
risk_placeholder = metric2.empty()
status_placeholder = metric3.empty()

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
else:
    uploaded_file = None

if mode == "Live Webcam":
    ctx = webrtc_streamer(
        key="opencv-filter",
        mode="sendrecv",
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 10}},
        video_frame_callback=lambda frame: None  # optional
    )
density_placeholder = metric1.empty()
risk_placeholder = metric2.empty()
status_placeholder = metric3.empty()

if 'live_processor' in locals():
    level, color = classify(live_processor.density)
    density_placeholder.markdown(
        f"<div class='card'><div class='metric-title'>Live Density</div>"
        f"<div class='metric-value' style='color:{color};'>{live_processor.density:.3f}</div></div>",
        unsafe_allow_html=True
    )
    risk_placeholder.markdown(
        f"<div class='card'><div class='metric-title'>Live Risk</div>"
        f"<div class='metric-value' style='color:{color};'>{level}</div></div>",
        unsafe_allow_html=True
    )
    status = "SAFE" if live_processor.density < 0.03 else "⚠️ OVERCROWDED"
    status_placeholder.markdown(
        f"<div class='card'><div class='metric-title'>Status</div>"
        f"<div class='metric-value' style='color:{color};'>{status}</div></div>",
        unsafe_allow_html=True
    )

chart = chart_container.line_chart([])

if 'densities' not in st.session_state:
    st.session_state.densities = []
densities = st.session_state.densities

# ==============================
# PROCESS VIDEO
# ==============================
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        density, mask = process_frame(frame)
        densities.append(density)

        level, color = classify(density)

        heatmap = generate_heatmap(mask)

        # VIDEO
        video_placeholder.image(frame, channels="BGR", use_container_width=True)

        # HEATMAP
        heatmap_placeholder.image(heatmap, channels="BGR", use_container_width=True)

        # METRICS
        density_placeholder.markdown(
            f"<div class='card'><div class='metric-title'>Density</div>"
            f"<div class='metric-value' style='color:{color};'>{density:.2f}</div></div>",
            unsafe_allow_html=True
        )

        risk_placeholder.markdown(
            f"<div class='card'><div class='metric-title'>Risk Level</div>"
            f"<div class='metric-value' style='color:{color};'>{level}</div></div>",
            unsafe_allow_html=True
        )

        status = "SAFE" if density < 0.03 else "OVERCROWDED"

        status_placeholder.markdown(
            f"<div class='card'><div class='metric-title'>Status</div>"
            f"<div class='metric-value' style='color:{color};'>{status}</div></div>",
            unsafe_allow_html=True
        )

        # GRAPH
        chart.add_rows([density])

    cap.release()

# ==============================
# DOWNLOAD
# ==============================
if densities:
    df = pd.DataFrame({"Density": densities})

    st.download_button(
        "Download Report",
        df.to_csv(index=False),
        "crowd_report.csv"
    )