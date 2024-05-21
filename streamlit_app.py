import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import re
from PIL import Image

# Load necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Setting header
st.title('Twitter Sentiment Analysis')
st.markdown("""
This application uses a pre-trained Passive Aggressive Classifier model to predict the sentiment of Twitter data. 
It also provides visualizations of the sentiment data.
""")

# Load the model and vectorizer
with open('pac_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define accuracy
accuracy_pac = 0.90

# Sidebar for user input
st.sidebar.header('User Input Features')
user_input = st.sidebar.text_area("Enter text for sentiment analysis:")

if st.sidebar.button("Predict Sentiment"):
    if user_input:
        # Transform the user input using the loaded TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Predict the sentiment using the loaded model
        prediction = model.predict(input_tfidf)

        # Display the prediction
        st.sidebar.write(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.sidebar.write("Please enter some text for analysis.")

# Display model accuracy
st.sidebar.write(f"Model Accuracy: {accuracy_pac * 100:.2f}%")

# Create columns for better layout
col1, col2 = st.columns(2)

# Training data visualization
with col1:
    st.subheader('Sentiment Training Data Visualization')

    # Read and display training WordCloud
    with open('all_text.txt', 'r', encoding='utf-8') as file:
        all_text = file.read()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud.to_array(), use_column_width=True)

# Testing data visualization
with col2:
    st.subheader('Sentiment Testing Data Visualization')

    # Read and display testing WordCloud
    with open('all_text1.txt', 'r', encoding='utf-8') as file:
        all_text1 = file.read()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text1)
    st.image(wordcloud.to_array(), use_column_width=True)

# Additional visualizations
st.subheader('Sentiment Distribution')

# Assuming you have data frames for visualization
# Replace with actual data if available
df_train = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Counts': [100, 80, 50]})
df_test = pd.DataFrame({'Sentiment': ['Positive', 'Negative', 'Neutral'], 'Counts': [30, 40, 30]})

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x='Sentiment', y='Counts', data=df_train, ax=ax[0])
ax[0].set_title('Training Data Sentiment Distribution')

sns.barplot(x='Sentiment', y='Counts', data=df_test, ax=ax[1])
ax[1].set_title('Testing Data Sentiment Distribution')

st.pyplot(fig)

# Optional feedback section
st.subheader('User Feedback')
feedback = st.text_area("Provide your feedback about the prediction:")
if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")

# Footer
st.markdown("""
*Created by Pawan Singh*

This project was inspired by the need for sentiment analysis in social media data. The sentiment analysis model was trained using a Passive Aggressive Classifier and TF-IDF vectorizer.
""")
