# prompt: build a streamlit.py file that will have the above data file . It will use vectorization method for conversion of string to numeric. Then will load  the random_forest.pkl file  and once we run it, will open a local host that will ask for user name, keyword as input and preferred job and salary will be as output
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon') # Download VADER lexicon for sentiment analysis
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('E:\\DigiCrome\\Summer Internship NextHikes\\Project-8\\Streamlit\\all_upwork_jobs_2024-02-07-2024-03-24.csv')
# Load the trained model
filename='E:\\DigiCrome\\Summer Internship NextHikes\\Project-8\\Streamlit\\random_forest_model (1).pkl'

model = pickle.load(open(filename, 'rb'))

# Load the vectorizer (important: use the same vectorizer used during training)
# You'll need to save the vectorizer during the training process
# Example:
# with open('vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

#Then load it here
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit app
st.title("Job Prediction App")

# Get user input
user_name = st.text_input("Enter your name:")
keyword = st.text_input("Enter a keyword:")
country = st.text_input("Enter the country")

# Combine inputs into a DataFrame (matching the format used for training)
input_data = pd.DataFrame({
    'title': [keyword],
    'country': [country]
})

# Combine text features
input_data['combined_text'] = input_data['title'] + ' ' + input_data['country']
input_data['combined_text'].fillna('', inplace=True)
input_data['combined_text'] = input_data['combined_text'].astype(str)

# Vectorize the input data
#X_input = vectorizer.transform(input_data['combined_text'][0])

# Make prediction
X_input = vectorizer.transform([input_data['combined_text'][0]]) # Assuming you combined text features
prediction = model.predict(X_input)[0] 

# Make predictions
#rediction = model.predict(X_input)[0]
#prediction=model.predict(X_input)[0]
# Example: Lower the threshold to 0.4
prediction = model.predict_proba(X_input)[:, 1] > 0.05
# Display the prediction
if st.button("Predict"):
    if prediction[0]:
        st.write(f"Hello {user_name}, based on the keyword '{keyword}' and country '{country}', this job is predicted to have high demand. ")

    else:
        st.write(f"Hello {user_name}, based on the keyword '{keyword}' and country '{country}', this job is not predicted to have high demand.")
