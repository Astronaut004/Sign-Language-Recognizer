from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading

app = Flask(__name__)
socketio = SocketIO(app)

detector = HandDetector(maxHands=1)
classifier = Classifier("./Model/keras_model.h5", "./Model/labels.txt")
offset = 20
imgSize = 300



labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "Bang Bang", "C", "Down", "Good Job",
          "I Love You", "Loser", "No", "UP", "Yes"]

# Flag to stop the video capture thread when the application is closed
stop_thread = False

def video_thread():
    cap = cv2.VideoCapture(0)

    while not stop_thread:
        success, img = cap.read()

        if success:
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                img_white = np.ones((300, 300, 3), np.uint8) * 255

                if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                    aspect_ratio = h / w

                    if aspect_ratio > 1:
                        k = 300 / h
                        w_cal = math.ceil(k * w)
                        img_resize = cv2.resize(img_crop, (w_cal, 300))
                        w_gap = int(math.ceil((300 - w_cal) / 2))
                        img_white[:, w_gap:w_gap + w_cal] = img_resize[:, :300]
                    else:
                        k = 300 / w
                        h_cal = math.ceil(k * h)
                        img_resize = cv2.resize(img_crop, (300, h_cal))
                        h_gap = int(math.ceil((300 - h_cal) / 2))
                        img_white[h_gap:h_gap + h_cal, :] = img_resize[:300, :]

                    prediction, index = classifier.getPrediction(img_white, draw=False)
                    result = {"prediction": labels[index]}
                    socketio.emit("update_prediction", result)

            # Convert the image to JPEG format and send it to the client
            _, buffer = cv2.imencode(".jpg", img)
            image_data = buffer.tobytes()
            socketio.emit("update_image", image_data)

    cap.release()

@app.route('/')
def index():
    return render_template('./index1.html')

if __name__ == '__main__':
    video_thread = threading.Thread(target=video_thread)
    video_thread.start()
    
    try:
        socketio.run(app, debug=True, use_reloader=False)
    finally:
        stop_thread = True
        video_thread.join() 

