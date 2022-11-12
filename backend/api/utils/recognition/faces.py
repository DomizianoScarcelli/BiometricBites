import numpy as np
import cv2

# Classifier
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # frontal face only

def test():
        return "Ciao :)"

def face_recognition():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Turn captured frame into gray scale
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the classifier to find faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        # Find the precise position of the faces
        for (x, y, w, h) in faces:
            # ROI: Region of interest, (ycord_start, ycord_end), the square in which the face was found
            #roi_gray = gray[y:y+h, x:x+w] 
            #roi_color = frame[y:y+h, x:x+w]

            # Saving detected face
            # img_item = "my-image.png"
            # cv2.imwrite(img_item, roi_gray)

            # Drawing a rectangle around the face
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): # Close the frame by pressing 'q'
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return