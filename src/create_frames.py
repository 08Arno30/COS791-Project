import cv2
import os

def create_frames():
    # Create a directory to store frames
    video_names = ['IceHockey', 'GirlsHockey']
    video_paths = ['../data/videos/Hockey0.mp4', '../data/videos/Hockey1.mp4-.mp4']

    for i in range(len(video_paths)):
        output_dir = f'../data/{video_names[i]}_extracted_frames'
        os.makedirs(output_dir, exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(video_paths[i])
        frame_count = 0

        # Extract frames
        while True:
            ret, frame = cap.read()

            # Break the loop if there are no more frames
            if not ret:
                break
            
            # Resize frame using ratio
            ratio = 0.68 if i == 0 else 1.75
            width = int(frame.shape[1] * ratio)
            height = int(frame.shape[0] * ratio)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            # Save frame as image
            cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames.")

    return video_names