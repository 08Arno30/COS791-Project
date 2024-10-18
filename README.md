# COS791-Project
In this project we implemented an Object Tracking algorithm by using a pre-trained YOLOv8 model and fine-tuning it on a few datasets, including a dataset that we had to manually label. Afterwards, we used the fine-tuned model to make predictions on detecting the hockey ball/puck in the two provided videos. Finally, we used the predicted boundary box's coordinates together with the built-in track method to track the object throughout the video and used a scaling factor of 1.5 to enlarge the detected ball region within each frame (distinct colours were also used for the bounding box to help clarify where the ball/puck is located).

### Environment Setup
1. Clone the Repository
2. Change working directory to `src`
3. Setup virtual environment </br>
    3.1 Run the command `pip install virtualenv` </br>
    3.2 Create env folder using the command `python -m venv env` </br>
    3.3 Activate the environment using the command `source env\\Scripts\\activate`. You should see the name of the env folder after running any command. </br>
    3.4 Install the required dependencies using the command `cd ./src && pip install -r requirements.txt`
4. Run the program using the command `python object_tracking.py`. This will generate an output video under `/data/output_videos`
5. When you are done, deactivate the virtual environment by running the command `deactivate`.

### Output Videos
Note the video links below were created by [Screenpal](https://go.screenpal.com) merely for demonstration purposes. ***The output videos were still created by our program.***

1. [Ice Hockey video](https://go.screenpal.com/watch/cZ6FirVWJMV)

2. [Field Hockey video](https://go.screenpal.com/watch/cZ6FirVWJMn)
