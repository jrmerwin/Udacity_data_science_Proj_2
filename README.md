# Udacity_data_science_Proj_2
Disaster Pipeline Project

# Disaster Response Pipeline Project

## Table of Contents 
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [File Descriptions](#descriptions)
4. [Licensing and Acknowledgements](#aknowledge)

## Introduction <a name=" introduction"></a>

This project uses a data set provided by *Figure Eight* to build a natural language processing pipeline to categorize social media messages during a natural disaster and provide a webapp interface to analyze new messages in real time. The webapp also visualizes some of the training data.

## Installation <a name=" introduction "></a>

Run the following commands in the project's root directory to set up your database and model. <br>

To run ETL pipeline that cleans data and stores in database <br>
<code> `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` </code> <br>
<br>To run ML pipeline that trains classifier and saves <br>
<code> `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` </code> <br>

Run the following command in the app's directory to run your web app. <br>
<code> `python run.py` </code>

Go to http://0.0.0.0:3001/ <br>

If the webapp doesn't load then you will need to install the files locally in a virtual environment and run it from the command prompt. Here is a link to the resource I used to figure out that work around.

 [https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

## File Descriptions <a name=" File Descriptions "></a>
<br>
1. process_data.py (ETL Pipeline)<br>
An ETL Pipeline Python script that writes a data cleaning pipeline that loads, cleans, and writes the training data to a SQLite Database. Located in the data folder
<br/>
2. train_classifier.py (ML Pipeline)<br>
This Python script writes a machine learning pipeline that loads data from the SQLite database, trains and optimizes a model using GridSearchCV, tests the model for accuracy, and saves the final model as a pickle file. Located in the model folder<br>
<br/>
3. app folder: Contains the files needed to load the model into the webapp and construct the app interface. <br>
<br/>

## Licensing and Acknowledgements <a name=" aknowledge "></a>
Disaster message dataset provided by [Figure Eight] (https://www.figure-eight.com/).
The instructions for executing the files and sections of the webapp html are materials from Udacity. I am NOT claiming to be the author on these materials!

