import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import pyrealsense2 as rs


def get_image(pipeline: rs.pipeline, align: rs.align) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns depth and color image from the camera.
    """
    frames = pipeline.wait_for_frames()     
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()  # Get depth frame.
    color_frame = frames.get_color_frame()  # Get color frame.
    
    # depth_frame = filtering_stage(depth_frame)    ## Use if you desire to filter the depth frame.

    # Transform images into numpy arrays.
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(color_frame.get_data())

    return depth_image, img

def filtering_stage(depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    Filters the depth frame in order to improve its quality. Use only if the code has been modified 
    to take into consideration the change in scale due to the decimation filter.
    """

    # Depth to disparity and disparity to depth transforms.
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # Decimation filter.
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 5)
    depth_frame = decimation.process(depth_frame)
    depth_frame = depth_to_disparity.process(depth_frame)

    # Spatial filter.
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    depth_frame = spatial.process(depth_frame)

    # Temporal filter.
    temporal = rs.temporal_filter()
    depth_frame = temporal.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)

    # Hole filling filter.
    hole_filling = rs.hole_filling_filter()
    depth_frame = hole_filling.process(depth_frame)

    return depth_frame

def object_detect(model: YOLO, depth_image: np.ndarray, img: np.ndarray, confidence: float,
                  classes: list[int], intr: rs.intrinsics) -> list[float]:
    """
    Identify the object (STOP SIGN) that will serve as the reference marker for the position
    estimation.
    """
    results = model(img, stream=True, conf=confidence, classes=classes) # Check the results from YOLO.

    # Iterate over the results.
    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])   # If the class identified belonged to the STOP sign...
            if cls == classes:

                # Compute median depth within the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                
                # Convert to meters and handle NaN values
                person_depth = depth_image[y1:y2, x1:x2]
                median_distance = np.nanmedian(person_depth) / 1000.0  

                x = int(np.mean([x1, x2]))
                y = int(np.mean([y1, y2]))

                # Deproject pixel into 3D point to obtain its coordinates.
                pos = rs.rs2_deproject_pixel_to_point(intr, [x, y], median_distance)
                x_coord = pos[0]    # X-coordinate
                y_coord = pos[2]    # Y-coordinate
                
                print(f'origin coordinates = ({x_coord:.2f}, {y_coord:.2f})')
                return [x_coord, y_coord]
    return []

def object_pose(model: YOLO, depth_image: np.ndarray, img: np.ndarray, confidence: float,
                classes: list[int], intr: rs.intrinsics, coord: list[float]) -> list[float]:
    """
    Identify the persons in the space and compute their relative transformed coordinates according
    to the plane axes of the machine to be monitored.
    """
    results = model(img, stream=True, conf=confidence, classes=classes) # Check the results from YOLO.
    
    # Iterate over the results.
    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])   # If the class identified belonged to Person...
            if cls == classes:
                # Compute median depth within the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                # Convert to meters and handle NaN values
                person_depth = depth_image[y1:y2, x1:x2]
                median_distance = np.nanmedian(person_depth) / 1000.0  

                x = int(np.mean([x1, x2]))
                y = int(np.mean([y1, y2]))

                # Deproject pixel into 3D point to obtain its coordinates.
                pos = rs.rs2_deproject_pixel_to_point(intr, [x, y], median_distance)
                x_coord = pos[0] - coord[0]
                y_coord = pos[2] - coord[1]

                # Rotation angle to adjust the reference plane.
                angle = np.pi/180.0*(-30.0)

                # Rotate coordinates to reference plane.
                x_coord_rot = np.cos(angle) * x_coord - np.sin(angle) * y_coord
                y_coord_rot = np.sin(angle) * x_coord + np.cos(angle) * y_coord

                
                print(f'student coordinates = ({x_coord_rot:.2f}, {y_coord_rot:.2f})')
                print("x1; ", x1, " x2: ", x2, " y1: ", y1, " y2: ", y2)

                # Check if the person's coordinates stay within the permitted bounds.
                if np.abs(y_coord_rot) < 0.90 and np.abs(x_coord_rot) < 1.70:
                    colorRect = (0, 0, 255) # Red
                else:
                    colorRect = (0, 255, 0) # Green

                # Draw the rectangle over the identified person and display their relative coordinates to the STOP sign.
                cv2.rectangle(img, (x1, y1), (x2, y2), colorRect, 3)
                cvzone.putTextRect(img, f'(x,y): ({x_coord_rot:.2f}, {y_coord_rot:.2f})',
                                   (max(0, x1-40), max(35, y1-40)), scale=1.5, thickness=1,
                                   colorR=colorRect)


if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Object to align color and depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Obtain intrinsic parameters of the camera.
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # YOLO model.
    model = YOLO('yolov8n.pt')

    origin = 11     # Class id for STOP sign.
    student = 0     # Class id for person.
    
    origin_coord = []
    
    # Wait until the reference marker is identified and located.
    while not origin_coord:
        depth_image, img = get_image(pipeline, align)
        origin_coord = object_detect(model, depth_image, img, confidence=0.50, classes=origin, intr=intr)
   
   # Continuously search for students in the image and display their coordinates.
    while True:
        depth_image, img = get_image(pipeline, align)
        object_pose(model, depth_image, img, confidence=0.50, classes=student, intr=intr, coord=origin_coord)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        
    pipeline.stop()
        

