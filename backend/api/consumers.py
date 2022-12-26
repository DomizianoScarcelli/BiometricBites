import json
from channels.generic.websocket import WebsocketConsumer
import cv2
from .utils.encoding.encoding import b64str_to_opencvimg, opencvimg_to_b64_str
from bsproject.settings import CLASSIFIER

class FrameConsumer(WebsocketConsumer):
    def connect(self):
       self.accept()
       self.send(text_data=json.dumps({
        "type": "connection_established",
        "message": "Your are now connected"
       }))
       self.count = 0
       self.classifier = CLASSIFIER
    
    def receive(self, text_data):
        self.count += 1
        try:
            img = b64str_to_opencvimg(text_data)
            processed_frame = self.classifier.recognize(img) if self.count % 3 == 0 else self.classifier.detect_faces(img)
            b64_img = opencvimg_to_b64_str(processed_frame)
            self.send(text_data=b64_img)
        except Exception as e:
            print("ERROR : "+str(e))