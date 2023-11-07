import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

model = YOLO('yolov8n.pt')

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(color_frame.get_data())
    
    results = model(img, stream=True, classes=0, conf=0.5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            if cls == 0:
                # Compute median depth within the bounding box
                person_depth = depth_image[y1:y2, x1:x2]
                median_distance = np.nanmedian(person_depth) / 1000.0  # Convert to meters and handle NaN values
                
                print("x1; ", x1, " x2: ", x2, " y1: ", y1, " y2: ", y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(img, f'Person Conf: {conf}', (max(0, x1), max(35, y1-20)), scale=1.5, thickness=1)
                cvzone.putTextRect(img, f'Distance: {median_distance:.2f}m', (max(0, x1), max(35, y1-40)), scale=1.5, thickness=1)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break