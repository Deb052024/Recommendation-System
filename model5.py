import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the training data and the trained model
df = pd.read_csv('E:\\DigiCrome\\Summer Internship NextHikes\\Project-8\\Streamlit\\all_upwork_jobs_2024-02-07-2024-03-24.csv')
filename = r'E:\\DigiCrome\\Summer Internship NextHikes\\Project-8\\Streamlit\\random_forest_model (1).pkl'  # Update with correct path if needed
model = pickle.load(open(filename, 'rb'))

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Job Prediction App")

user_name = st.text_input("Enter your name:")
keyword = st.text_input("Enter a keyword:")
country = st.text_input("Enter the country:")
budget = st.text_input("Enter the budget:")  # Added budget input

input_data = pd.DataFrame({'title': [keyword], 'country': [country], 'budget': [budget]})

input_data['combined_text'] = input_data['title'] + ' ' + input_data['country']
input_data['combined_text'].fillna('', inplace=True)
input_data['combined_text'] = input_data['combined_text'].astype(str)

if st.button("Predict"):
    X_input = vectorizer.transform([input_data['combined_text'][0]])
    prediction_proba = model.predict_proba(X_input)[0]
    high_demand_prob = prediction_proba[1]
    median_salary = df['budget'].median()  # Adjust column name if needed
    if high_demand_prob > 0.03 and float(budget) > median_salary:  # Convert budget to float
        st.markdown(f"<p style='color:green;'>Hello {user_name}, based on the job title '{keyword}', country '{country}', and salary, this job is predicted to be in **high demand**.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:yellow;'>Hello {user_name}, based on the job title '{keyword}', country '{country}', and salary, this job is predicted to be in **low demand**.</p>", unsafe_allow_html=True)