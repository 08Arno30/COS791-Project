import cv2
import os

# Create a directory to store frames
video_name = ['IceHockey', 'GirlsHockey']
video_path = ['./videos/Hockey0.mp4', './videos/Hockey1.mp4-.mp4']

for i in range(len(video_path)):
    output_dir = f'{video_name[i]}_extracted_frames'
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path[i])
    frame_count = 0

    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")
