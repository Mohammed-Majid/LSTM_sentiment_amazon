import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from langdetect import detect
from googletrans import Translator

# Function to load model and tokenizer
def load_resources():
    try:
        model = load_model('sentiment_analysis_model.h5')
        with open('tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load the model and tokenizer
model, tokenizer = load_resources()

# Fixed settings
MAX_LENGTH = 100  # Example fixed length
THRESHOLD = 0.5   # Example fixed threshold

# Initialize Translator
translator = Translator()

def detect_and_translate(review_text):
    try:
        language = detect(review_text)
        if language != 'en':
            translation = translator.translate(review_text, src=language, dest='en').text
            return translation, language
        return review_text, language
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return review_text, 'en'

def predict_sentiment(review_text):
    sequences = tokenizer.texts_to_sequences([review_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH)
    prediction = model.predict(padded_sequences)
    confidence = prediction[0][0]
    sentiment = "Positive" if confidence > THRESHOLD else "Negative"
    return sentiment, confidence

# History management
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(review_text, sentiment, confidence, language=None, translated_text=None):
    st.session_state.history.append((review_text, sentiment, confidence, language, translated_text))

# Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter a review text to get its sentiment prediction.')

# Text input from user
review_text = st.text_area("Review Text", "")

# Predict button
if st.button('Predict Sentiment'):
    if review_text:
        try:
            translated_text, language = detect_and_translate(review_text)
            sentiment, confidence = predict_sentiment(translated_text)
            st.write(f"Predicted Sentiment: {sentiment}")
            st.write(f"Confidence Score: {confidence:.2f}")
            if language != 'en':
                st.write(f"Original Language: {language}")
                st.write(f"Translated Review: {translated_text}")
            add_to_history(review_text, sentiment, confidence, language, translated_text)
        except Exception as e:
            st.write(f"Error during prediction: {e}")
    else:
        st.write("Please enter a review text.")

# Add a collapsible section for history
with st.expander("View Prediction History"):
    if st.session_state.history:
        for review_text, sentiment, confidence, language, translated_text in st.session_state.history:
            st.write(f"**Review:** {review_text}")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
            if language and language != 'en':
                st.write(f"**Original Language:** {language}")
                st.write(f"**Translated Review:** {translated_text}")
            st.write("---")
    else:
        st.write("No history available.")
