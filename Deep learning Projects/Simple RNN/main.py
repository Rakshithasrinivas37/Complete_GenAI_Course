import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

import streamlit as st

## load the imdb dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

## Load the model

model = load_model('simple_rnn_model.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Predict function

def predict(review):
    preprocessed_input = preprocess(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

## streamlit app
st.title("Movie review sentiment analysis")

st.write("Enter a movie review: ")

#Take user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess(user_input)

    sentiment, score = predict(preprocessed_input)

    #Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Predicted score: {score}')
else:
    st.write('Please enter a movie review')

    