import requests

def send_to_remote(url, bbox_list, center_list):

    payload = {
        "bounding_boxes": bbox_list,
        "center_points": center_list
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code
    except:
        return "Failed"
