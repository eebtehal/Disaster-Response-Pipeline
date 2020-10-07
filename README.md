# Disaster Response Pipeline Project
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) pipeline automating categorizing messages on a real time basis.


### Description
A machine learning pipeline to categorize emergency messages based on the needs communicated by the sender and Web App to show model results in real time.


### Files in the repository
* data (process_data.py: cleans data and stores in database)
* models (train_classifier.py: trains classifier and saves the model)
* app (run.py: runs the web dashboard)


### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
* [https://www.udacity.com/][Udacity] for providing such a complete Data Science Nanodegree Program
* [Figure Eight] [https://appen.com/] for providing messages dataset to train my model
