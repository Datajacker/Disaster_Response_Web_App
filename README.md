# Disaster_Response_Web_App
This app utilized the dataset from Figure Eight and built up a natural language process pipeline to analyze the massage and help disaster response.

# Project Motivation
After a disaster, there will be thousand of messages on social media. It is essential for disaster response organizations to analyze those messages and deliver what is needed to those people. Different organization could take care of different responsiblities, such as medical supplies, food, etc. Supervised machine learning models can help extract the words and direct messages to those organizations.

# Installation
A few python libraries are used for this web app:
* Numpy
* Pandas
* re
* sys
* sklearn
* sqlalchemy
* nltk

# Instruction to run
## Udacity workpalce
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Open another terminal and type `env|grep WORK` this will give you the spaceid (it will start with view*** and some characters after that)

4. Now open your browser window and type https://view6914b2f4-3001.udacity-student-workspaces.com/go?query=water, replace the whole **view6914b2f4** with your space id that you got in the step 3

5. Press enter and the app should now run for you. You can try out with different messages.

## Local machine
1. Once your app is running (python run.py)
2. Go to http://localhost:3001 and the app will now run


# File description
The file structure is shown as the folowing:

# How to interact with this project
You can send pull request or suggestions to me.
