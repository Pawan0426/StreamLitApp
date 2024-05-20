import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import time, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import re

#Setting header
st.header('Twitter Sentiment Analysis')

#Setting subheader
st.subheader('Sentiment training data visizulation.')

# Read text data from the file
with open('all_text.txt', 'r', encoding='utf-8') as file:
    all_text = file.read()

# Generate the WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the WordCloud using Streamlit
st.image(wordcloud.to_array(), use_column_width=True)

#Setting subheader
st.subheader('Sentiment testing data visizulation.')

# Read text data from the file
with open('all_text1.txt', 'r', encoding='utf-8') as file:
    all_text1 = file.read()

# Generate the WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the WordCloud using Streamlit
st.image(wordcloud.to_array(), use_column_width=True)

# Load the model and vectorizer
with open('pac_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


accuracy_pac = 0.90

# Streamlit app
st.title('Sentiment Analysis Using Passive Aggressive Classifier')

# Input text from user
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict Sentiment"):
    if user_input:
        # Transform the user input using the loaded TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Predict the sentiment using the loaded model
        prediction = model.predict(input_tfidf)

        # Display the prediction
        st.write(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.write("Please enter some text for analysis.")

# Optional: Display the accuracy of the model
st.write(f"Model Accuracy: {accuracy_pac}")

