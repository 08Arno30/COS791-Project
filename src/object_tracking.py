import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort from deep_sort_realtime

video_names = ['Hockey0.mp4', 'Hockey1.mp4-.mp4']
selected_video = video_names[1]

video_path = f'../data/original_video_frames_and_labels/videos/{selected_video}'

# # Create output directory
# if not os.path.exists('../data/output_videos'):
#     os.makedirs('../data/output_videos', exist_ok=True)

# video_out_path = os.path.join('../data/output_videos', f'{selected_video.split(".")[0]}_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))

# Load fine-tuned YOLO model
model = YOLO("../fine_tuned_model/train/weights/best.pt")

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()
