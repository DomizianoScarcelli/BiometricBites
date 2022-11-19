import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import cv2
from .utils.encoding.encoding import b64str_to_opencvimg, opencvimg_to_b64_str

class FrameConsumer(WebsocketConsumer):
    def connect(self):
       self.accept()
       self.send(text_data=json.dumps({
        "type": "connection_established",
        "message": "Your are now connected"
       }))
    
    def receive(self, text_data):
        img = b64str_to_opencvimg(text_data)
        cv2.line(img,(0,0),(511,511),(255,0,0),5)
        b64_img = opencvimg_to_b64_str(img)
        self.send(text_data=b64_img)

