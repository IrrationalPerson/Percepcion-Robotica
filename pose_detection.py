import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import pyrealsense2 as rs


def get_image(pipeline, align):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    # depth_frame = filtering_stage(depth_frame)

    color_frame = frames.get_color_frame()
    
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(color_frame.get_data())

    return depth_image, img


def object_pose(model, depth_image, img, confidence, classes, intr, coord):
    results = model(img, stream=True, conf=confidence, classes=classes)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            obj_conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            # for c in classes:
            if cls == classes:
                # Compute median depth within the bounding box
                person_depth = depth_image[y1:y2, x1:x2]
                median_distance = np.nanmedian(person_depth) / 1000.0  # Convert to meters and handle NaN values

                x = int(np.mean([x1, x2]))
                y = int(np.mean([y1, y2]))

                # pos = rs.rs2_deproject_pixel_to_point(intr, [x, y], median_distance)
                x_coord = median_distance * (x - intr.ppx) / intr.fx - coord[0]
                y_coord = median_distance * (y - intr.ppy) / intr.fy - coord[1]
                z_coord = median_distance - coord[2]
                
                print(f'Orange coordinates = ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f})')
                # print(median_distance)
                # print(depth_image.shape)
                # print(img.shape)


                # x = int(np.mean([x1, x2]))
                # y = int(np.mean([y1, y2]))
                # median_distance = depth_image[int(y/5), int(x/5)] / 1000.0


                print("x1; ", x1, " x2: ", x2, " y1: ", y1, " y2: ", y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(img, f'Person Conf: {obj_conf}', (max(0, x1-40), max(35, y1-20)), scale=1.5, thickness=1)
                # cvzone.putTextRect(img, f'Distance: {median_distance:.2f}m', (max(0, x1-40), max(35, y1-40)), scale=1.5, thickness=1)
                cvzone.putTextRect(img, f'(x,y,z): ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f})', (max(0, x1-40), max(35, y1-40)), scale=1.5, thickness=1)


def object_detect(model, depth_image, img, confidence, classes, intr):
    results = model(img, stream=True, conf=confidence, classes=classes)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # obj_conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            # for c in classes:
            if cls == classes:
                # Compute median depth within the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                
                person_depth = depth_image[y1:y2, x1:x2]
                median_distance = np.nanmedian(person_depth) / 1000.0  # Convert to meters and handle NaN values

                x = int(np.mean([x1, x2]))
                y = int(np.mean([y1, y2]))

                # pos = rs.rs2_deproject_pixel_to_point(intr, [x, y], median_distance)
                x_coord = median_distance * (x - intr.ppx) / intr.fx
                y_coord = median_distance * (y - intr.ppy) / intr.fy
                z_coord = median_distance
                
                print(f'Apple coordinates = ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f})')
                # print(x1, y1, x2, y2)
                return [x_coord, y_coord, z_coord]
    return []

# filters
def filtering_stage(depth_frame):
    #depth to disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # Decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 5)
    depth_frame = decimation.process(depth_frame)
    depth_frame = depth_to_disparity.process(depth_frame)

    # Spatial filter
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    depth_frame = spatial.process(depth_frame)


    # Temporal filter
    temporal = rs.temporal_filter()
    depth_frame = temporal.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)

    # Hole Filling
    hole_filling = rs.hole_filling_filter()
    depth_frame = hole_filling.process(depth_frame)

    return depth_frame


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)


    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    model = YOLO('yolov8n.pt')

    apple = 47
    orange = 49
    
    apple_coord = []
    
    while not apple_coord:
        depth_image, img = get_image(pipeline, align)
        apple_coord = object_detect(model, depth_image, img, confidence=0.05, classes=apple, intr=intr)

    # x1 = detect_apple[0]
    # y1 = detect_apple[1]
    # x2 = detect_apple[2]
    # y2 = detect_apple[3]

    # x = int(np.mean(x1, x2))
    # y = int(np.mean(y1, y2))
   

    while True:
        
        depth_image, img = get_image(pipeline, align)
        object_pose(model, depth_image, img, confidence=0.15, classes=orange, intr=intr, coord=apple_coord)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        

    pipeline.stop()
        

