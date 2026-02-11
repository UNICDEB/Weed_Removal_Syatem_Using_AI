import socket
import json

# CHANGE THIS to receiver IP
RECEIVER_IP = "192.168.1.20"
PORT = 5000


def send_to_remote(bbox_list, center_list):

    payload = {
        "bounding_boxes": bbox_list,
        "center_points": center_list
    }

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((RECEIVER_IP, PORT))

        json_data = json.dumps(payload)
        client.sendall(json_data.encode())

        client.close()

        print("Data sent successfully to receiver.")
        return "Success"

    except Exception as e:
        print("Error sending data:", e)
        return "Failed"
