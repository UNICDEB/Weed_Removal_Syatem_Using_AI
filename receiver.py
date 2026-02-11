import socket
import json
from datetime import datetime

# -------- CONFIG --------
HOST = "0.0.0.0"   # Listen on all interfaces
PORT = 5000
# ------------------------

# Storage variables
received_bounding_boxes = []
received_center_points = []

def start_server():
    global received_bounding_boxes, received_center_points

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)

    print(f"Receiver running on port {PORT}...")
    print("Waiting for data...\n")

    while True:
        conn, addr = server.accept()
        print(f"Connected from {addr}")

        try:
            data = conn.recv(4096).decode()

            if not data:
                continue

            json_data = json.loads(data)

            received_bounding_boxes = json_data.get("bounding_boxes", [])
            received_center_points = json_data.get("center_points", [])

            print("\n========== DATA RECEIVED ==========")
            print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Bounding Boxes Coordinates:")
            print(received_bounding_boxes)
            print("Center Point Coordinates:")
            print(received_center_points)
            print("===================================\n")

        except Exception as e:
            print("Error:", e)

        finally:
            conn.close()


if __name__ == "__main__":
    start_server()
