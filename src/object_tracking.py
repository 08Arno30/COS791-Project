import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort from deep_sort_realtime

video_names = ['Hockey0.mp4', 'Hockey1.mp4-.mp4']
selected_video = video_names[1]

video_path = f'../data/original_video_frames_and_labels/videos/{selected_video}'

# Create output directory
if not os.path.exists('../data/output_videos'):
    os.makedirs('../data/output_videos', exist_ok=True)

video_out_path = os.path.join('../data/output_videos', f'{selected_video.split(".")[0]}_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# Load fine-tuned YOLO model
model = YOLO("../fine_tuned_model/train/weights/best.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=0, n_init=5, nms_max_overlap=0.5) # parameters might need to be tuned to optimize the tracker

# Set bounding box color for detection
bbox_color = (0, 255, 0)

detection_threshold = 0.7

while ret:
    # Run object detection using YOLO
    # resize the frame to 640x640
    frame = cv2.resize(frame, (640, 640))
    results = model(frame)
    
    for result in results:
        detections = []
        for bbox in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = bbox

            # convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # apply detection threshold
            if score > detection_threshold:
                detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_id])
        
        # Update object tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id  # track id assigned by DeepSORT
            ltrb = track.to_ltrb()  # bounding box in format (left, top, right, bottom)
            
            # Ensure the bounding box coordinates are valid integers
            x1, y1, x2, y2 = map(int, ltrb)

            # Check if the bounding box is within the frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

    # Display frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Write frame to output video
    cap_out.write(frame)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
