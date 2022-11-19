import numpy as np
import base64
import cv2

def b64str_to_opencvimg(b64_str):
    b64_str = b64_str.replace("data:image/jpeg;base64,", "")
    jpg_original = base64.b64decode(b64_str)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img

def opencvimg_to_b64_str(opencvimg):
    _, encoded_im_arr = cv2.imencode('.jpeg', opencvimg)
    im_bytes = encoded_im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    string = im_b64.decode('utf-8')
    string = "data:image/jpeg;base64," + string
    return string