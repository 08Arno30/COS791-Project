import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # Import DeepSort from deep_sort_realtime

video_names = ['Hockey0.mp4', 'Hockey1.mp4-.mp4']
selected_index = 1
selected_video = video_names[selected_index]

video_path = f'../data/original_video_frames_and_labels/videos/{selected_video}'

# Create output directory
if not os.path.exists('../data/output_videos'):
    os.makedirs('../data/output_videos', exist_ok=True)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

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

        # save annotated_frame to a folder
        cv2.imwrite('../data/output_videos/frame_' + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))) + '.jpg', annotated_frame)

        # Display the annotated frame
        # cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# retrieve frames if it is .jpg
def get_frames(output_dir):
    frames = []
    for file in os.listdir(output_dir):
        if file.endswith('.jpg'):
            frames.append(file)
    
    # sort the frame names according the sequence
    frames = sorted(frames, key=lambda x: int(x.split('.')[0].split('_')[1]))
    return frames
output_frames = get_frames('../data/output_videos')

# save frames as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

video_names = ['Hockey0_out.mp4', 'Hockey1_out.mp4']
video_out_path = f'../data/output_videos/{video_names[selected_index]}'
cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (640, 640))

for frame_file in output_frames:
    file_path = '../data/output_videos/' + frame_file
    frame = cv2.imread(file_path)
    frame = cv2.resize(frame, (640, 640))
    cap_out.write(frame)

    # remove the file
    os.remove(file_path)

cap_out.release()
cap.release()
cv2.destroyAllWindows()