# ======================================================================
# This file is only used to convert the video into frames and label them
# ======================================================================

import create_frames
import label_images
import os


if __name__ == '__main__':
    # Check if data is available
    if os.path.exists('../data') and os.path.exists('../data/videos') and len(os.listdir('../data')) - 1 != len(os.listdir('../data/videos'))*2:
        print('Data is not available. Extracting data from videos...')
        file_names  = create_frames.create_frames()

        print('Creating labels...')
        label_images.label_images(file_names)

