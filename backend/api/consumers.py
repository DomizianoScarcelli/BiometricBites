import json
from channels.generic.websocket import WebsocketConsumer
import cv2
from .utils.encoding.encoding import b64str_to_opencvimg, opencvimg_to_b64_str
from bsproject.settings import CLASSIFIER
from .utils.user_info.user_info import get_user_info
import numpy as np

class FrameConsumer(WebsocketConsumer):
    def connect(self):
       self.accept()
       self.send(text_data=json.dumps({
        "type": "connection_established",
        "message": "Your are now connected"
       }))
       self.count = 0
       self.classifier = CLASSIFIER
       self.DELTA_RECOGNITION = 5
    
    def receive(self, text_data):
        self.count += 1
        identity_data = {}
        # {
        #   FRAME: str
        #   RECOGNITION_PHASE: boolean
        #   FACE_PRESENT: boolean
        #   USER_INFO: None | 
        # {
        #     ID: int,
        #     NAME: str,
        #     SURNAME: str,
        #     CF: str,
        #     COST: int
        # }
        #   SIMILARITY: str
        #     
        # }
        if text_data == "null":
            return 
        img = b64str_to_opencvimg(text_data)
        processed_frame = None
        if self.count % self.DELTA_RECOGNITION == 0:
            processed_frame, id, similarity = self.classifier.recognize(img)
            identity_data["FACE_PRESENT"] = id is not None
            identity_data["USER_INFO"] = get_user_info(id) if id is not None else None
            identity_data["RECOGNITION_PHASE"] = True
        else:
            processed_frame, face_present = self.classifier.detect_faces(img)
            similarity = 1.0
            identity_data["FACE_PRESENT"] = face_present
            identity_data["RECOGNITION_PHASE"] = False
        b64_img = opencvimg_to_b64_str(processed_frame)
        identity_data["FRAME"] = b64_img
        identity_data["SIMILARITY"] = np.float64(similarity)
        identity_data = json.dumps(identity_data)
        self.send(text_data=identity_data)