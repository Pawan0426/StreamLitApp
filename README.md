# Twitter Sentiment Analysis App

This Streamlit web application performs sentiment analysis on Twitter data using a Passive Aggressive Classifier. It includes features for data visualization and sentiment prediction.

## Overview

This app is designed to analyze sentiment in Twitter data using a pre-trained model. It provides visualizations of sentiment distributions in both training and testing datasets, and allows users to input text for real-time sentiment prediction.

## Features

- **WordCloud Visualization:** The app generates WordCloud visualizations of sentiment data from both training and testing datasets.
- **Real-time Sentiment Prediction:** Users can input text for sentiment analysis and receive real-time predictions using the pre-trained model.
- **Model Accuracy Display:** The app displays the accuracy of the sentiment analysis model.

## Setup and Usage

1. **Install Dependencies:** Make sure you have the required dependencies installed. You can install them using `pip`:

   ```bash
   pip install streamlit pandas numpy altair matplotlib seaborn wordcloud nltk
   
**Clone Repository**: Clone this repository to your local machine:

```bash
git clone <repository-url>

**Run the App**: Navigate to the project directory and run the Streamlit app:

```bash
streamlit run app.py

## Files and Structure
- **app.py:** Contains the main code for the Streamlit web application.
- **pac_model.pkl:** Pre-trained model file for sentiment analysis.
- **tfidf_vectorizer.pkl:** Pickle file containing TF-IDF vectorizer for text preprocessing.
- **all_text.txt:** Text file containing training dataset for sentiment analysis.
- **all_text1.txt:** Text file containing testing dataset for sentiment analysis.

  
## Technologies Used
- **Streamlit**: For building the interactive web application.
- **Pandas, NumPy**: For data manipulation and processing.
- **Altair, Matplotlib, Seaborn**: For data visualization.
- **NLTK**: For natural language processing tasks.
- **WordCloud**: For generating WordCloud visualizations.

## Acknowledgements
This project was inspired by the need for sentiment analysis in social media data. The sentiment analysis model was trained using a Passive Aggressive Classifier and TF-IDF vectorizer.

## Author
Pawan Singh
