import os
import cv2
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
camera_thread = None
running = False


# ===========================
# CAMERA LIVE THREAD
# ===========================

def camera_loop():
    global running
    camera.start()

    while running:
        color, depth, _ = camera.get_aligned_frames()
        if color is not None:
            cv2.imshow("Live RGB 1280x720", color)
            cv2.waitKey(1)

    camera.stop()
    cv2.destroyAllWindows()


# ===========================
# ROUTES
# ===========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/open_camera")
def open_camera():
    global running, camera_thread
    running = True
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.start()
    return {"status": "Camera Opened"}


@app.post("/start_process")
def start_process(conf: float = 0.5):

    global latest_result

    color, depth_image, depth_frame = camera.get_aligned_frames()

    annotated, boxes = detector.detect(color, conf)

    formatted_results = []

    for box in boxes:
        x1, y1, x2, y2, conf_score, cls = box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Center depth
        center_depth, center_3d = pixel_to_3d(
            depth_frame,
            camera.intrinsics,
            cx, cy
        )

        # Convert meter to cm
        center_3d_cm = [round(i * 100, 2) for i in center_3d]

        # Bounding box corners 3D
        bbox_depth, bbox_3d = pixel_to_3d(
            depth_frame,
            camera.intrinsics,
            x1, y1
        )

        bbox_3d_cm = [round(i * 100, 2) for i in bbox_3d]

        formatted_results.append({
            "bbox_pixel": [x1, y1, x2, y2],
            "bbox_coordinate_cm": bbox_3d_cm,
            "center_pixel": [cx, cy],
            "center_coordinate_cm": center_3d_cm
        })

    # Save image
    filename = f"{SAVE_FOLDER}/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, annotated)

    latest_result = {
        "image": filename,
        "detections": formatted_results
    }

    return latest_result



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
