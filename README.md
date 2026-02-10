# 🌿 Weed Detection System

The **Weed Detection System** is a real-time AI vision system built using **Intel RealSense D435i**, **YOLO (COCO Pretrained Model)**, and **FastAPI**.  
It detects weeds from RGB frames, aligns depth data, converts pixel positions into real-world 3D coordinates (in cm), and displays structured results in a modern web dashboard.

---

## 🚀 Features

- Real-time RGB (1280x720) + Depth Alignment
- YOLO Object Detection with Confidence Slider
- Depth Filtering Pipeline:
  - Decimation (Magnitude = 1)
  - Depth → Disparity
  - Spatial Filter
  - Temporal Filter
  - Disparity → Depth
- Pixel → 3D Coordinate Conversion (cm)
- Separate Conversion for:
  - Bounding Box Starting Pixel (x1, y1)
  - Bounding Box Ending Pixel (x2, y2)
  - Center Pixel (cx, cy)
- Automatic Result Image Saving
- Structured Output Formatting
- External Device Data Transmission (API)
- Light-Themed Web Dashboard
- Threaded Camera Processing

---

## 📦 Project Structure

```
Weed_Detection_System/
│
├── main.py
├── config.py
├── requirements.txt
├── camera/
├── detection/
├── utils/
├── static/result/
└── templates/index.html
```

---

## 📐 Output Format

### Bounding Box Pixel
```
Starting Pixel: [x1, y1]
Ending Pixel: [x2, y2]
```

### Bounding Box 3D Coordinate (cm)
```
Starting Coordinate: [X1, Y1, Z1]
Ending Coordinate: [X2, Y2, Z2]
```

### Center Coordinate (cm)
```
Center Pixel: [cx, cy]
Center Coordinate: [Xc, Yc, Zc]
```

### Internal Data Structure
```python
[
  [sx, sy, sz, ex, ey, ez],
  ...
]
```

---

## ⚙️ Installation

```bash
python -m venv weedenv
weedenv\Scripts\activate
pip install -r requirements.txt
```

For CPU-only PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Run the System

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 🔄 Workflow

1. Open Camera  
2. Adjust Confidence Slider  
3. Click Start  
4. System:
   - Captures aligned RGB + Depth
   - Detects object
   - Converts pixels to 3D coordinates (cm)
   - Saves result image
   - Displays formatted output
   - Sends coordinates to external device (optional)

---

## 🛠️ Technologies

FastAPI • PyRealSense2 • OpenCV • Ultralytics YOLO • Bootstrap 5 • REST API

---

## 👨‍💻 Developed By

Debabrata Doloi 
AI Vision & Robotics Developer  

---

**Weed Detection System – Real-Time AI for Smart Agriculture**
