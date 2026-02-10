# 🌿 Weed Detection System

An AI-based Real-Time Weed Detection System developed using **Intel RealSense D435i**, **YOLO (COCO Pretrained Model)**, and **FastAPI Web Framework**.

This system captures aligned RGB and Depth frames, performs object detection, converts pixel coordinates to real-world 3D coordinates, and displays formatted detection results in a modern web dashboard.

---

## 🚀 Features

- ✅ Real-time RGB (1280x720) and Depth Alignment
- ✅ YOLO Object Detection (COCO Pretrained)
- ✅ Confidence Threshold Control (Dynamic Slider)
- ✅ Bounding Box Pixel Extraction:
  - Starting Pixel (x1, y1)
  - Ending Pixel (x2, y2)
- ✅ Pixel → Real World Coordinate Conversion (cm)
- ✅ Separate Conversion for:
  - Starting Pixel → 3D Coordinate
  - Ending Pixel → 3D Coordinate
  - Center Pixel → 3D Coordinate
- ✅ Depth Filtering Pipeline:
  - Decimation Filter (Magnitude = 1)
  - Depth to Disparity
  - Spatial Filter
  - Temporal Filter
  - Disparity to Depth
- ✅ Save Detection Result Image Automatically
- ✅ Send Bounding Box and Center Coordinates to External Device
- ✅ Web-Based Dashboard (Light Theme)
- ✅ Processing Animation Indicator
- ✅ Threaded Camera Handling for Performance

---

## 🏗️ Project Structure

Weed_Detection_System/
│
├── main.py
├── config.py
├── requirements.txt
│
├── camera/
│ ├── realsense_camera.py
│ ├── depth_processing.py
│
├── detection/
│ ├── yolo_detector.py
│
├── utils/
│ ├── coordinate_utils.py
│ ├── sender.py
│
├── static/
│ ├── result/
│
└── templates/
└── index.html


---

## ⚙️ System Requirements

- Python 3.9 / 3.10 (Recommended)
- Intel RealSense D435i
- Windows 64-bit
- Microsoft Visual C++ Redistributable (2015–2022)
- GPU (Optional – CPU works fine)

---

## 📦 Installation

### 1️⃣ Create Virtual Environment

```bash
python -m venv weedenv
weedenv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

If using CPU-only PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


▶️ Run the System
uvicorn main:app --reload

Open browser:
http://127.0.0.1:8000

🧠 Detection Workflow

Click Open Camera

Adjust Confidence Slider

Click Start

System:

Captures aligned color + depth frame

Applies depth filters

Detects object using YOLO

Extracts:

Bounding Box Pixels

Center Pixel

Converts to 3D Real-World Coordinates (cm)

Saves result image

Sends data to external device (if configured)

Displays formatted results in dashboard

📐 Output Format
Bounding Box Pixel Format
Starting Pixel: [x1, y1]
Ending Pixel: [x2, y2]

Bounding Box Coordinate Format (cm)
Starting Coordinate: [X1, Y1, Z1]
Ending Coordinate: [X2, Y2, Z2]

Center Coordinate Format (cm)
Center Pixel: [cx, cy]
Center Coordinate: [Xc, Yc, Zc]


Internal Data Structure
[
  [sx, sy, sz, ex, ey, ez],
  [sx, sy, sz, ex, ey, ez],
  ...
]


🔄 Depth Processing Order

Decimation Filter (Magnitude = 1)

Depth → Disparity

Spatial Filter

Temporal Filter

Disparity → Depth

Ensures smoother and stable depth estimation.

💾 Saved Results

Detected images are automatically saved in:
static/result/

Format:
result_YYYYMMDD_HHMMSS.jpg


🌐 API Endpoints
| Endpoint         | Method | Description             |
| ---------------- | ------ | ----------------------- |
| `/`              | GET    | Web Dashboard           |
| `/open_camera`   | POST   | Starts RealSense Camera |
| `/start_process` | POST   | Runs Detection          |
| `/clear`         | POST   | Clears Results          |
| `/exit`          | POST   | Stops Camera            |

🔗 External Device Integration

Coordinates can be transmitted to another system via FastAPI lightweight API service using:
{
  "bounding_boxes": [...],
  "center_points": [...]
}

Used for:

Robotic Actuation

Industrial Automation

Smart Agriculture Systems

🛠️ Technologies Used

FastAPI

OpenCV

PyRealSense2

Ultralytics YOLO

Bootstrap 5

Threading

REST API Communication

👨‍💻 Developed By

Debabrata Doloi
AI Vision & Robotics Developer

📌 Future Enhancements

Live Video Streaming in Browser (WebSocket)

Multi-object Tracking

3D Visualization Dashboard

CSV Export

Robotic Arm Direct Integration

ROS2 Compatibility

Cloud Deployment

📄 License

This project is developed for research and industrial automation purposes.