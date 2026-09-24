import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import cast
import cv2 as cv
import numpy as np
import time

#Just a small setup for the face tracker, probably will have to change a lot to be able to implement it into the project

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode
LatestResult = None
# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print('face detector result: {}'.format(result))
    global LatestResult 
    LatestResult = result

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/tudor/Desktop/Project 2-1/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with FaceDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
    cap = cv.VideoCapture(0)

    #

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
  

   
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while cap.isOpened():

        ret, frame = cap.read()

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) 
        detector.detect_async(mp_image, time.monotonic_ns() // 1_000_000)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if LatestResult is not None:
            h, w = frame.shape[:2]
            for detection in LatestResult.detections:
                bbox = detection.bounding_box
                cv.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                             (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                             (0, 255, 0), 2, 16)
                for kp in detection.keypoints:
                    cv.circle(frame, (int(kp.x * w), int(kp.y * h)), 4, (0, 0, 255), -1)



        frame = cv.flip(frame, 1)
        # Display the resulting frame

        
        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        
        
    
    cap.release()
    out.release()
    cv.destroyAllWindows()