"""
AetherEdge Vehicle Counting & Flow Analysis Dashboard
Three-state lifecycle: READY → RUNNING → FINISHED
Flask server with MJPEG stream, SSE alerts, CSV/JSON export, annotated video download.
"""
import os
import time
import json
import threading
import queue
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, jsonify,
    request, send_file, stream_with_context
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from tracker import Sort
from counter import VehicleCounter
from detector import VehicleDetector

app = Flask(__name__)

# ──────────────────────────────────────────
# Folders
# ──────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
DATASET_VIDEO = os.path.join(BASE_DIR, "datasets", "video", "cars.mp4")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ──────────────────────────────────────────
# Global state
# status: "ready" | "running" | "finished"
# ──────────────────────────────────────────
state = {
    "video_path":        DATASET_VIDEO,
    "video_name":        "cars.mp4",
    "conf_thresh":       0.42,
    "line_coords":       [[220, 420], [1080, 420]],
    "direction":         "top_to_bottom",
    "status":            "ready",   # <-- lifecycle state
    "fps":               50.0,
    "processing_fps":    0.0,
    "frame_skip":        1,
    "total_frames":      0,
    "current_frame":     0,
    "output_video_path": os.path.join(OUTPUT_FOLDER, "result_cars.mp4"),
    "video_duration":    0.0,
}

frame_lock  = threading.Lock()
sse_clients = []
sse_lock    = threading.Lock()

# "Ready" splash frame
_blank = np.zeros((540, 960, 3), dtype=np.uint8)
cv2.putText(_blank, "READY — Press  Start Analysis", (180, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
cv2.putText(_blank, "Configure the counting line, then click Start.", (180, 310),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
_, _jpeg_init = cv2.imencode(".jpg", _blank)
current_jpeg_frame = _jpeg_init.tobytes()

# ──────────────────────────────────────────
# AI singletons
# ──────────────────────────────────────────
detector = VehicleDetector(confidence_threshold=state["conf_thresh"])
tracker  = Sort(max_age=30, min_hits=3, iou_threshold=0.35)
counter  = VehicleCounter(line_coords=state["line_coords"],
                          target_direction=state["direction"])

CLASS_COLORS = {
    "car":        (46,  204, 113),
    "truck":      (241, 196,  15),
    "bus":        (155,  89, 182),
    "motorcycle": ( 52, 152, 219),
    "other":      (149, 165, 166),
}

# ──────────────────────────────────────────
# SSE broadcast
# ──────────────────────────────────────────
def broadcast(payload: dict):
    msg = f"data: {json.dumps(payload)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


# ──────────────────────────────────────────
# Background processing loop
# ──────────────────────────────────────────
def video_processing_loop():
    global state, detector, tracker, counter, current_jpeg_frame, frame_lock

    print("[Engine] Processing loop started.")
    cap    = None
    writer = None
    current_path = None

    frame_idx         = 0
    fps_frame_counter = 0
    last_fps_check    = time.time()
    cached_tracks     = np.empty((0, 6))

    while True:
        # ── READY / FINISHED: show splash, do nothing ──
        if state["status"] != "running":
            time.sleep(0.1)
            continue

        # ── Open video when starting a new run ──
        if current_path != state["video_path"] or cap is None or not cap.isOpened():
            if cap    is not None: cap.release()
            if writer is not None: writer.release()

            current_path = state["video_path"]
            cap = cv2.VideoCapture(current_path)
            if not cap.isOpened():
                print(f"[Error] Cannot open: {current_path}")
                state["status"] = "ready"
                time.sleep(0.5)
                continue

            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            state["fps"]            = orig_fps
            state["total_frames"]   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            state["video_duration"] = state["total_frames"] / orig_fps

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            out_name = f"result_{state['video_name']}"
            state["output_video_path"] = os.path.join(OUTPUT_FOLDER, out_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(state["output_video_path"], fourcc, orig_fps, (w, h))

            frame_idx     = 0
            cached_tracks = np.empty((0, 6))
            tracker.__init__(max_age=30, min_hits=3, iou_threshold=0.35)
            tracker.frame_count = 0
            print(f"[Engine] Opened: {current_path}  {w}x{h} @ {orig_fps:.1f} fps")

        # ── Read frame ──
        t_start = time.time()
        ret, frame = cap.read()

        # ── Video finished naturally ──
        if not ret:
            if writer is not None:
                writer.release()
                writer = None
            cap.release()
            cap = None
            current_path = None
            state["status"] = "finished"
            print(f"[Engine] Video complete. Total crossings: {counter.counts['total']}")
            # Broadcast finished event to UI
            broadcast({"type": "finished", "total": counter.counts["total"]})
            continue

        frame_idx += 1
        fps_frame_counter += 1
        state["current_frame"] = frame_idx

        # Compute processing FPS
        now = time.time()
        if now - last_fps_check >= 1.0:
            state["processing_fps"] = round(fps_frame_counter / (now - last_fps_check), 1)
            fps_frame_counter = 0
            last_fps_check    = now

        current_ts = frame_idx / state["fps"]

        # ── Detect + track ──
        if frame_idx % state["frame_skip"] == 0 or len(cached_tracks) == 0:
            dets, cls_names = detector.detect(frame)
            if len(dets) > 0:
                tracked_objs, _ = tracker.update(dets, cls_names)
                cached_tracks = tracked_objs
            else:
                cached_tracks, _ = tracker.update(np.empty((0, 5)), [])
                if len(cached_tracks) == 0:
                    cached_tracks = np.empty((0, 6))

        # ── Line crossing ──
        if len(cached_tracks) > 0:
            new_events = counter.process_tracks(
                cached_tracks,
                video_name=state["video_name"],
                current_timestamp=current_ts,
            )
            for ev in new_events:
                broadcast(ev)

        # ── Annotate frame ──
        annotated = frame.copy()
        lc = state["line_coords"]
        p1 = (int(lc[0][0]), int(lc[0][1]))
        p2 = (int(lc[1][0]), int(lc[1][1]))

        cv2.line(annotated, p1, p2, (255, 0, 128), 3, cv2.LINE_AA)
        cv2.circle(annotated, p1, 7, (0, 255, 255), -1)
        cv2.circle(annotated, p2, 7, (0, 255, 255), -1)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        cv2.putText(annotated, "COUNT LINE", (mid[0]-50, mid[1]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        for obj in cached_tracks:
            x1,y1,x2,y2 = int(float(obj[0])),int(float(obj[1])),int(float(obj[2])),int(float(obj[3]))
            tid   = int(obj[4])
            cname = str(obj[5]).lower() if len(obj)>5 else "car"
            color = CLASS_COLORS.get(cname, (0,255,0))
            box_color = (0,255,128) if tid in counter.counted_ids else color
            cv2.rectangle(annotated, (x1,y1), (x2,y2), box_color, 2)
            label = f"#{tid} {cname.upper()}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            cv2.rectangle(annotated, (x1, max(0,y1-20)), (x1+tw+8,y1), box_color, -1)
            cv2.putText(annotated, label, (x1+4, max(12,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 2, cv2.LINE_AA)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(annotated, (cx,cy), 4, (0,255,255), -1)
            for trk in tracker.trackers:
                if trk.id == tid and len(trk.centroid_history) > 1:
                    pts = trk.centroid_history
                    for i in range(1, len(pts)):
                        a = i / len(pts)
                        tc = tuple(int(c*a) for c in color)
                        cv2.line(annotated, pts[i-1], pts[i], tc, 2, cv2.LINE_AA)

        # HUD
        total = counter.counts["total"]
        ov = annotated.copy()
        cv2.rectangle(ov, (12,12),(310,72),(10,15,35),-1)
        annotated = cv2.addWeighted(ov, 0.7, annotated, 0.3, 0)
        cv2.rectangle(annotated,(12,12),(310,72),(255,0,128),2)
        cv2.putText(annotated, f"VEHICLES CROSSED: {total}", (22,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

        # Progress bar
        if state["total_frames"] > 0:
            prog  = frame_idx / state["total_frames"]
            bw    = annotated.shape[1] - 24
            bh    = annotated.shape[0]
            cv2.rectangle(annotated, (12,bh-18),(12+bw,bh-6),(30,30,50),-1)
            cv2.rectangle(annotated, (12,bh-18),(12+int(bw*prog),bh-6),(255,0,128),-1)
            ts_str = f"{current_ts:.1f}s / {state['video_duration']:.1f}s"
            cv2.putText(annotated, ts_str, (annotated.shape[1]-165, bh-22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)

        if writer is not None:
            writer.write(annotated)

        ret_enc, jpeg = cv2.imencode(".jpg", annotated,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ret_enc:
            with frame_lock:
                current_jpeg_frame = jpeg.tobytes()

        elapsed = time.time() - t_start
        delay   = 1.0 / state["fps"]
        if elapsed < delay:
            time.sleep(delay - elapsed)


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frame")
def get_frame():
    with frame_lock:
        data = current_jpeg_frame
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                             "Pragma": "no-cache"})


@app.route("/api/stream")
def mjpeg_stream():
    """True MJPEG stream — one persistent connection, no rapid polling needed."""
    def generate():
        boundary = b"--frame"
        while True:
            with frame_lock:
                data = current_jpeg_frame
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   data + b"\r\n")
            # Push at ~15fps — enough for smooth display without hammering the server
            time.sleep(0.067)
    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/alerts/stream")
def alert_stream():
    q = queue.Queue(maxsize=100)
    with sse_lock:
        sse_clients.append(q)

    def event_stream():
        try:
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stats")
def get_stats():
    progress = (state["current_frame"] / state["total_frames"] * 100
                if state["total_frames"] > 0 else 0)
    return jsonify({
        "status":          state["status"],
        "video_name":      state["video_name"],
        "processing_fps":  state["processing_fps"],
        "video_fps":       state["fps"],
        "progress_pct":    round(progress, 1),
        "current_frame":   state["current_frame"],
        "total_frames":    state["total_frames"],
        "video_duration":  round(state["video_duration"], 1),
        "counts":          counter.counts,
        "total_crossings": counter.counts["total"],
        "cars":            counter.counts["car"],
        "trucks":          counter.counts["truck"],
        "buses":           counter.counts["bus"],
        "motorcycles":     counter.counts["motorcycle"],
        "top_to_bottom":   counter.counts["top_to_bottom"],
        "bottom_to_top":   counter.counts["bottom_to_top"],
    })


@app.route("/api/events")
def get_events():
    return jsonify(counter.events)


# ── Lifecycle controls ──
@app.route("/api/start", methods=["POST"])
def start_analysis():
    """Start or resume processing."""
    if state["status"] == "finished":
        # Re-run: reset counter and re-open video
        counter.reset()
        state["current_frame"] = 0
    state["status"] = "running"
    print("[API] Analysis started.")
    return jsonify({"status": "running"})


@app.route("/api/stop", methods=["POST"])
def stop_analysis():
    """Pause / stop processing without discarding results."""
    state["status"] = "ready"
    print("[API] Analysis stopped.")
    return jsonify({"status": "ready"})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file attached"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    # Reset everything for the new video
    state["status"]        = "ready"
    state["video_path"]    = save_path
    state["video_name"]    = f.filename
    state["current_frame"] = 0
    counter.reset()

    # Reset splash frame
    global current_jpeg_frame
    splash = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(splash, f"Loaded: {f.filename}", (60, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,200), 2)
    cv2.putText(splash, "Configure the line, then click  Start Analysis", (60, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    _, j = cv2.imencode(".jpg", splash)
    with frame_lock:
        current_jpeg_frame = j.tobytes()

    print(f"[Upload] New video loaded: {save_path}")
    return jsonify({"status": "ready", "video_name": f.filename})


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.json or {}
    if "conf_thresh"  in data:
        state["conf_thresh"] = float(data["conf_thresh"])
        detector.set_confidence(state["conf_thresh"])
    if "line_coords"  in data:
        state["line_coords"] = data["line_coords"]
        counter.update_line(state["line_coords"], state["direction"])
    if "direction"    in data:
        state["direction"] = data["direction"]
        counter.update_line(state["line_coords"], state["direction"])
    if "frame_skip"   in data:
        state["frame_skip"] = max(1, int(data["frame_skip"]))
    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
def reset_all():
    state["status"]        = "ready"
    state["current_frame"] = 0
    counter.reset()
    return jsonify({"status": "ready"})


@app.route("/api/export_csv")
def export_csv():
    return Response(
        counter.get_csv_data(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=vehicle_counts_{state['video_name']}.csv"},
    )


@app.route("/api/export_json")
def export_json():
    summary = counter.get_json_summary(duration_sec=state["video_duration"])
    return Response(
        json.dumps(summary, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition":
                 f"attachment; filename=summary_{state['video_name']}.json"},
    )


@app.route("/api/download_video")
def download_video():
    path = state["output_video_path"]
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return send_file(path, mimetype="video/mp4", as_attachment=True,
                         download_name=f"annotated_{state['video_name']}")
    return jsonify({"error": "Annotated video not ready yet."}), 404


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=video_processing_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    print(f"[AetherEdge] Dashboard -> http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
