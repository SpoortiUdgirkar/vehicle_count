"""
AetherEdge - Vehicle Counting & Flow Analysis
Streamlit Interface for Deployment
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import streamlit as st
import threading
import cv2
import time
import numpy as np
import pandas as pd

from detector import VehicleDetector
from tracker import Sort
from counter import VehicleCounter

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AetherEdge Vehicle Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 100%); }
    h1 { color: #ff50a0 !important; }
    .metric-card {
        background: rgba(255,80,160,0.1);
        border: 1px solid rgba(255,80,160,0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Global State (persists across Streamlit reruns) ───────────
if "_vc_state" not in st.session_state:
    st.session_state["_vc_state"] = {
        "status": "ready",
        "total": 0, "cars": 0, "trucks": 0, "buses": 0,
        "frame_count": 0, "total_frames": 1,
    }
if "_vc_events" not in st.session_state:
    st.session_state["_vc_events"] = []
if "_vc_frame" not in st.session_state:
    st.session_state["_vc_frame"] = None
if "_vc_thread" not in st.session_state:
    st.session_state["_vc_thread"] = None

VIDEO_PATH = os.path.join("datasets", "video", "cars.mp4")

# ── Detector (cached — loads once) ────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_detector():
    return VehicleDetector(confidence_threshold=0.42)

# ── Processing Loop ───────────────────────────────────────────
def processing_loop(state_ref, events_ref, frame_holder):
    detector = load_detector()
    tracker  = Sort(max_age=30, min_hits=2, iou_threshold=0.25)
    counter  = VehicleCounter(line_p1=(220, 420), line_p2=(1080, 420))

    cap = cv2.VideoCapture(VIDEO_PATH)
    state_ref["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    state_ref["status"] = "running"

    frame_idx = 0
    while cap.isOpened() and state_ref["status"] == "running":
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 2 != 0:          # skip every other frame for speed
            continue

        # Detect + Track + Count
        dets, cls_names = detector.detect(frame)
        tracks = tracker.update(dets if len(dets) > 0 else np.empty((0, 5)))
        crossings = counter.update(tracks, cls_names, dets)

        for c in crossings:
            state_ref["total"] += 1
            cls = c.get("class", "car")
            if cls == "car":      state_ref["cars"]   += 1
            elif cls == "truck":  state_ref["trucks"] += 1
            elif cls == "bus":    state_ref["buses"]  += 1
            events_ref.append({
                "Time":      c.get("timestamp", ""),
                "Track ID":  c.get("track_id", ""),
                "Class":     cls,
                "Direction": c.get("direction", ""),
            })

        state_ref["frame_count"] = frame_idx

        # ── Annotate ──────────────────────────────────────────
        vis = frame.copy()
        cv2.line(vis, (220, 420), (1080, 420), (0, 255, 100), 2)
        for t in tracks:
            x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (100,255,100), 2)
            cv2.putText(vis, f"#{int(tid)}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
        cv2.putText(vis, f"VEHICLES CROSSED: {state_ref['total']}",
                    (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Store as RGB for Streamlit
        frame_holder[0] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        time.sleep(0.05)

    cap.release()
    state_ref["status"] = "finished"


# ── UI ────────────────────────────────────────────────────────
st.markdown("# 🚗 AetherEdge Vehicle Counter")
st.caption("Real-time vehicle detection · SORT tracking · Line-crossing counter")
st.divider()

state   = st.session_state["_vc_state"]
events  = st.session_state["_vc_events"]

# KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("🎯 Total Vehicles",  state["total"])
k2.metric("🚗 Cars",            state["cars"])
k3.metric("🚛 Heavy Vehicles",  state["trucks"] + state["buses"])
k4.metric("📊 Status",          state["status"].upper())

st.divider()

# Controls
b1, b2, b3, _ = st.columns([1, 1, 1, 4])

if state["status"] in ("ready", "finished"):
    if b1.button("▶ Start Analysis", type="primary", use_container_width=True):
        # Reset
        state.update({"status": "starting", "total": 0, "cars": 0,
                      "trucks": 0, "buses": 0, "frame_count": 0})
        events.clear()
        frame_holder = [None]
        st.session_state["_vc_frame_holder"] = frame_holder
        t = threading.Thread(
            target=processing_loop,
            args=(state, events, frame_holder),
            daemon=True
        )
        t.start()
        st.session_state["_vc_thread"] = t
        st.rerun()

elif state["status"] in ("running", "starting"):
    if b1.button("⏹ Stop", type="secondary", use_container_width=True):
        state["status"] = "ready"
        st.rerun()

if b2.button("🔄 Reset", use_container_width=True):
    state.update({"status": "ready", "total": 0, "cars": 0,
                  "trucks": 0, "buses": 0, "frame_count": 0})
    events.clear()
    st.session_state["_vc_frame_holder"] = [None]
    st.rerun()

# Progress bar
progress = state["frame_count"] / max(state["total_frames"], 1)
st.progress(min(progress, 1.0),
            text=f"Frame {state['frame_count']} / {state['total_frames']}")

st.divider()

# Video + Events
vid_col, evt_col = st.columns([3, 2])

with vid_col:
    st.subheader("Live Feed")
    frame_ph = st.empty()
    fh = st.session_state.get("_vc_frame_holder", [None])
    if fh[0] is not None:
        frame_ph.image(fh[0], channels="RGB", use_column_width=True)
    else:
        frame_ph.info("📹 Press **Start Analysis** to begin")

with evt_col:
    st.subheader(f"Crossing Events ({len(events)})")
    if events:
        df = pd.DataFrame(events[-30:])
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No crossings detected yet")

# ── Auto-refresh while running ────────────────────────────────
if state["status"] in ("running", "starting"):
    time.sleep(0.4)
    st.rerun()

if state["status"] == "finished":
    st.success(f"✅ Analysis complete!  Total vehicles counted: **{state['total']}**")
    st.balloons()
