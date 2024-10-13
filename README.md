# COS791-Project

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
