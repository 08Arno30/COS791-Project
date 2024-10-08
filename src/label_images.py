import cv2
import os

# Function to save YOLO format labels
def save_yolo_label(img_shape, bbox, label_file):
    h, w, _ = img_shape
    x_min, y_min, width, height = bbox

    # Convert to YOLO format (normalize the bounding box coordinates)
    x_center = (x_min + (x_min + width)) / 2.0 / w
    y_center = (y_min + (y_min + height)) / 2.0 / h
    norm_width = width / w
    norm_height = height / h

    # Label format: <class-id> <x_center> <y_center> <width> <height>
    label = f"0 {x_center} {y_center} {norm_width} {norm_height}\n"

    # Save to a file
    with open(label_file, 'w') as f:
        f.write(label)

def label_images(file_names):
    # Folder paths
    for file_name in file_names:
        if not os.path.exists(f'../data/{file_name}_labels'):
            os.makedirs(f'../data/{file_name}_labels')
        
        frames_dir = f'../data/{file_name}_extracted_frames'
        labels_dir = f'../data/{file_name}_labels'

        print(f"Labeling frames in {frames_dir}...")

        # sort the frame names according the sequence
        frame_names = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('.')[0].split('_')[1]))

        # Loop over the extracted frames and label them
        for i, frame_name in enumerate(frame_names):
            # check if there is already a label for this image
            if os.path.exists(os.path.join(labels_dir, frame_name.replace('.jpg', '.txt'))):
                continue
            print(f"Labeling frame {i+1}/{len(frame_names)}...")
            # create empty txt file for frame
            with open(os.path.join(labels_dir, frame_name.replace('.jpg', '.txt')), 'w') as f:
                pass

            if frame_name.endswith('.jpg'):
                frame_path = os.path.join(frames_dir, frame_name)

                # Read the image
                img = cv2.imread(frame_path)

                # Use cv2.selectROI to select the ball (puck) region
                bbox = cv2.selectROI("Select ROI", img, showCrosshair=True)

                satisfied = False

                # prompt user to confirm selection or redo selection
                while not satisfied:
                    user_input = input("Are you satisfied with the selection? (y/n): ")

                    if user_input.lower() == 'y':
                        satisfied = True
                    elif user_input.lower() == 'n':
                        satisfied = False
                        # Redo selection
                        cv2.destroyWindow("Select ROI")
                        bbox = cv2.selectROI("Select ROI", img, showCrosshair=True)

                # If a bounding box is selected (non-zero width and height)
                if bbox[2] > 0 and bbox[3] > 0:
                    # Save the YOLO format label
                    label_path = os.path.join(labels_dir, frame_name.replace('.jpg', '.txt'))
                    save_yolo_label(img.shape, bbox, label_path)
                    print(f"Saved label for {frame_name}")

                # Close the ROI window
                cv2.destroyWindow("Select ROI")

label_images(['IceHockey', 'GirlsHockey'])