# Disaster Response Pipeline Project
### Disaster Response Pipeline Project
This Project is part of Data Science Nanodegree Program by Udacity in collaboration Figure Eight. The dataset contains pre-labelled messages from real-life disaster The project aim is to build a Natural Language Processing (NLP) pipeline automating categorizing messages on a real time basis.

### Description
A machine learning pipeline to categorize emergency mes. nes based on the needs communicated by the sender and Web App to show model result, in real time.

### Project Motivation
The goal of this project is to demonstrate usability of classification model in context of disaster information.Usage of this classifier can make easier decision where particular message should be sent to make a good action e.g. when message contains information about food or medicines are required, then this message can be sent to parties connected supplying food/medicines.



### Files in the repository 
* data (process_data.py: cieans data and stores in database) 
* models (train_classifier.py: trains classifier and saves the model) 
* app (run.py: runs the web dashboard)

### Repo Files:
The project has the following file structure:
```
workspace
├── README.md
├── app
|   ├── run.py  # Flask file that runs app
|   └── templates
|       ├──go.html      # classification result page of web app
|       └──master.html  # main page of web app
├── data
|   ├── disaster_categories.csv # data to process
|   ├── disaster_messages.csv   # data to process
|   ├── DisasterResponse.db     # database to save clean data to
|   └── process_data.py
└── models
    ├── train_classifier.py
    └── classifier.pkl  # saved model
    
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements 
* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program 
* [Figure Eight](https:///appen.com/) for providing messages dataset to train my model
