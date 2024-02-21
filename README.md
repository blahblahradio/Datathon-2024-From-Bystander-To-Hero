# MDA-Project: Noise in Leuven

## Main Files and Folders Description:
- *create_venv.ps1*              : PowerShell script to create a virtual environment using the requirements.txt file
- *requirements.txt*             : Contains the required packages for the project
- *unzipping_the_data.py*       : Python script to unzip and arrange all the data in csv format. 
- *prepping_the_data.py*        : Python script to generate required data for the analysis and app
- *app.py*                       : Python script that contains the main structure and body of the Dash app
- *assets/*                      : Contains images and CSS styles used in the Dash app
- *pages/*                       : Contains the Python scripts for the pages in the Dash app
- *optimization_notebooks/*      : Contains Jupyter notebooks concerning the various optimization algorithms

## How to start the project locally:
After cloning the repository:
- Run the *create_venv.ps1* file on the terminal to create the virtual environment (if on mac or linux, create the venv manually and run *pip install -r requirements.txt*).
- Run the full *unzipping_the_data.py* to obtain ready-to-use csv files with the project data. This has to be ran before both the instructions below.
- Run the *prepping_the_data.py* to obtain the required data for the analysis and app (takes some time).