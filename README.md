# Democratizing the prediction of student's dropout: The Interactive Visual DashBoard tool

# Requirements
  - Python 3.9 or Python 3.10

# Installation
  1. Download this repository or clone it using `git clone https://github.com/0Kan0/Democratizing-the-prediction-of-student-s-dropout-The-Interactive-Visual-DashBoard-tool.git`.
  2. Install the required packages by running `pip install -r requirements.txt`.
  3. Once the installation has ended, access to where the libraries are installed, access explainerdashboard library folder and open dashboards.py. There you should change this line of code in order for this app     to work:
  <p align="center">
  <img alt="Before" src="images/Before.png" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="After" src="images/After.png" width="45%">
  </p>

# Usage
  4. Access src folder and run `python index.py`. This will open your web browser and redirect to the home screen. If it doesn't, navigate to http://localhost:8080.
  5. Follow the instrunctions provided in the home screen on the file before uploading it.
  6. Upload the file.
  7. Press "Start" buttton.
  8. Wait for the process to finish. You can check it in the terminal (note that depending of the size of the dataset and how good your computer is, the process will take more or less). The AutoML model of the dataset will be saved in the [saved_AutoML_models](saved_AutoML_models/). If the same dataset is uploaded, the model will be loaded instead of training a new one.
  9. If all went correctly, a "Go to dashboard" button should appear below the "Start" button. Click it and that will open a new tab with the dashboard hub (in case it didn't redirect, navigate to                  http://127.0.0.1:8050).
  10. If you want load a new dataset, go to the terminal and end the process with Ctrl+C. Then open it again with `python index.py`.

# Home page
![dashboard.gif](images/home_page.gif)

# Dashboard hub
![dashboard.gif](images/dashboard_hub.gif)

There will be 2 dashboards we can see in the hub:
  - AutoML Student Dropout Explainer (Basic Interface): Focused for those that have no knowledge about machine learning. In this dashboard you can acces the "Predictions" tab and "What If" tab.
  - AutoML Student Dropout Explainer (Advanced Interface): Focused for those that have knowledge about machine learning. In addition of what was already in the other dashboard, here you can all models that were tested by AutoML and which one was the best, as well as different graphs and metrics of the performance of the best model used.
# Acknowledgements
