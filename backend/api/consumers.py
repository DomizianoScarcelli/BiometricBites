import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
import cv2
from .utils.encoding.encoding import b64str_to_opencvimg, opencvimg_to_b64_str
from .utils.Recognition.recognition import recognize

class FrameConsumer(WebsocketConsumer):
    def connect(self):
       self.accept()
       self.send(text_data=json.dumps({
        "type": "connection_established",
        "message": "Your are now connected"
       }))
    
    def receive(self, text_data):
        try:
            img = b64str_to_opencvimg(text_data)
            processed_frame = recognize(img)
            b64_img = opencvimg_to_b64_str(processed_frame)
            self.send(text_data=b64_img)
        except:
            print("No photo")
        
       

