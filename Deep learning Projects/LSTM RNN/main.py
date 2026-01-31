import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the model
model = load_model('lstm_model.h5')

## load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

#Predict the next function
def predict_next_word(model, tokenizer, text, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):] #To ensure the sequence length matches max_seq_len
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    prediction = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(prediction, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Word Prediction with LSTM")
input_text = st.text_input("Enter your sequence: ")

if st.button("Predict"):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_seq_len)
    st.write(f"Nex word: {next_word}")
