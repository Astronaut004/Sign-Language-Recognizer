import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)

# to detect hand in images
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "NewData/0"


counter=0

# webcam
while True:
    success, img = cap.read()

    # hand detected
    hands, img = detector.findHands(img)

    # Crop the image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # matrix
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        # Check if imgCrop is not empty before resizing
        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                # Resize if aspect ratio is greater than 1 (height > width)
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = int(math.ceil((imgSize - wCal) / 2))
                # Ensure imgResize is within the bounds of imgSize
                imgWhite[:, wGap:wGap + wCal] = imgResize[:, :imgSize]

            else:
                # Resize if aspect ratio is less than or equal to 1 (width >= height)
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = int(math.ceil((imgSize - hCal) / 2))
                # Ensure imgResize is within the bounds of imgSize
                imgWhite[hGap:hGap + hCal, :] = imgResize[:imgSize, :]

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
