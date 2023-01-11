import json
from channels.generic.websocket import WebsocketConsumer
import cv2
from .utils.encoding.encoding import b64str_to_opencvimg, opencvimg_to_b64_str
from bsproject.settings import CLASSIFIER
from .utils.user_info.user_info import get_user_info, get_profile_pic
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

    #TODO: this switch can be avoided if the load_labels and load_recognizer are defined, even abstractly, inside the generic Classifier
       # reload labels and classifier
       if self.classifier.name == "LBPHF":
        #LBPH
        self.classifier.labels = self.classifier.load_labels()
        self.classifier.load_recognizer()
       elif self.classifier.name == "SVC":
        #SVC
        self.classifier.labels = self.classifier.load_labels()
        self.classifier.classifier = self.classifier.load_classifier()
       elif self.classifier.name == "VGGFACE":
        #VGGFACE
        pass
    
    def receive(self, text_data):
        self.count += 1
        identity_data = {}
        # {
        #   STATE: "UNKNOWN", "KNOWN", "NO FACE" 
        #   FRAME: str
        #   RECOGNITION_PHASE: boolean (Is time for the recognition because of the delta or not)
        #   USER_INFO: None | 
        # {
        #     ID: int,
        #     NAME: str,
        #     SURNAME: str,
        #     CF: str,
        #     COST: int
        #     PROFILE_IMG: str
        # }
        #   SIMILARITY: str
        #     
        # }

        # If ID is None, then it means there is no face
        # If ID is "unknown", then the face isn't recognized
        # Otherwise the ID will be the ID of the recognized user

        if text_data == "null":
            return 
        img = b64str_to_opencvimg(text_data)
        processed_frame = None
        if self.count % self.DELTA_RECOGNITION == 0:
            processed_frame, id, similarity = self.classifier.recognize(img)
            identity_data["RECOGNITION_PHASE"] = True
            identity_data["FACE_PRESENT"] = id is not None
            if id is None:
                identity_data["STATE"] = "NO FACE"
                identity_data["FRAME"] = text_data
                identity_data = json.dumps(identity_data)
                self.send(text_data=identity_data)
                return
            if id.lower() == "unknown":
                identity_data["STATE"] = "UNKNOWN"
                identity_data = json.dumps(identity_data)
                self.send(text_data=identity_data)
                return
            identity_data["STATE"] = "KNOWN"
            identity_data["USER_INFO"] = get_user_info(id)
            identity_data["SIMILARITY"] = np.float64(similarity)
            identity_data["USER_INFO"]["PROFILE_IMG"] = get_profile_pic(id)
        else:
            processed_frame, face_present = self.classifier.detect_faces(img)
            identity_data["RECOGNITION_PHASE"] = False
            identity_data["FACE_PRESENT"] = face_present
        b64_img = opencvimg_to_b64_str(processed_frame)
        identity_data["FRAME"] = b64_img
        identity_data = json.dumps(identity_data)
        self.send(text_data=identity_data)