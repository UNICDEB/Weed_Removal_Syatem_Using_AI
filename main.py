import os
import cv2
import numpy as np
import threading
import io
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.sender import send_to_remote

from config import *
from camera.realsense_camera import RealSenseCamera
from detection.yolo_detector import YOLODetector
from utils.coordinate_utils import pixel_to_3d
from utils.sender import send_to_remote

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

camera = RealSenseCamera()
detector = YOLODetector(MODEL_PATH)

latest_result = {}
latest_frame = None
camera_thread = None
running = False
frame_lock = threading.Lock()
last_encoded_frame = None
last_encoded_time = 0


# ===========================
# CAMERA LIVE THREAD
# ===========================

def camera_loop():
    global running, latest_frame
    camera.start()
    frame_count = 0

    while running:
        color, depth, _ = camera.get_aligned_frames()
        if color is not None:
            frame_count += 1
            # Only update frame every other iteration to reduce encoding load
            if frame_count % 1 == 0:
                with frame_lock:
                    latest_frame = color.copy()

    camera.stop()


def generate_frames():
    global latest_frame
    while running:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            else:
                continue
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')
        else:
            continue


# ===========================
# ROUTES
# ===========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/open_camera")
def open_camera():
    global running, camera_thread
    if not running:
        running = True
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
    return {"status": "Camera Opened"}


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_frame")
def video_frame():
    """Return a single frame as JPEG - ultra-fast 1280x720 streaming"""
    global latest_frame
    
    if latest_frame is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank, [
            cv2.IMWRITE_JPEG_QUALITY, 20,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
    else:
        with frame_lock:
            frame = latest_frame.copy()
        
        # Ultra-aggressive compression for 30 FPS @ 1280x720
        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 20,  # Minimum quality
            cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimize JPEG structure
        ])
    
    if ret:
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close"
            }
        )
    return {"error": "Failed to encode frame"}


@app.post("/start_process")
def start_process(conf: float = 0.5):
    global latest_frame

    with frame_lock:
        if latest_frame is None:
            return {"error": "No camera frame available"}
        color = latest_frame.copy()

    # Get depth frame
    color_check, depth_image, depth_frame = camera.get_aligned_frames()

    annotated, boxes = detector.detect(color, conf)

    detection_results = []
    bbox_coordinate_list = []   # [[sx,sy,sz,ex,ey,ez],...]

    for box in boxes:

        x1, y1, x2, y2, conf_score, cls = box

        # -------- STARTING PIXEL -> 3D ----------
        start_depth, start_3d = pixel_to_3d(
            depth_frame,
            camera.intrinsics,
            x1, y1
        )

        # -------- ENDING PIXEL -> 3D ----------
        end_depth, end_3d = pixel_to_3d(
            depth_frame,
            camera.intrinsics,
            x2, y2
        )

        # Convert meter to cm
        start_3d_cm = [round(i * 100, 2) for i in start_3d]
        end_3d_cm = [round(i * 100, 2) for i in end_3d]

        bbox_coordinate_list.append(
            start_3d_cm + end_3d_cm
        )

        # -------- CENTER ----------
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        center_depth, center_3d = pixel_to_3d(
            depth_frame,
            camera.intrinsics,
            cx, cy
        )

        center_3d_cm = [round(i * 100, 2) for i in center_3d]

        detection_results.append({
            "bbox_pixel": {
                "starting_pixel": [x1, y1],
                "ending_pixel": [x2, y2]
            },
            "bbox_coordinate_cm": {
                "starting_coordinate": start_3d_cm,
                "ending_coordinate": end_3d_cm
            },
            "center_pixel": [cx, cy],
            "center_coordinate_cm": center_3d_cm
        })

    # Save image
    filename = f"{SAVE_FOLDER}/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, annotated)

    # ======================================
    # SEND DATA TO RECEIVER (ADD HERE)
    # ======================================
    if len(bbox_coordinate_list) > 0:
        # Extract center list separately
        center_coordinate_list = [
            d["center_coordinate_cm"]
            for d in detection_results
        ]

        send_to_remote(
            bbox_coordinate_list,
            center_coordinate_list
        )

    return {
        "image": filename,
        "detections": detection_results,
        "bbox_coordinate_list": bbox_coordinate_list
    }



@app.post("/close_camera")
def close_camera():
    global running
    running = False
    return {"status": "Camera Closed"}


@app.post("/clear")
def clear():
    global latest_result
    latest_result = {}
    return {"status": "Cleared"}


@app.post("/exit")
def exit_app():
    global running
    running = False
    return {"status": "Stopped"}
