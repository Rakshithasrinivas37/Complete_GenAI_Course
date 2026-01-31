import streamlit as st
import pandas as pd
import numpy as np

st.title("Basic AI Application")

project_name = st.text_input("Enter your project name: ")
st.write(f"Your project name is {project_name}")

batch_size = st.slider("Enter the Batch size number:", 0, 256, 2)
st.write(f"Selected batch size is {batch_size}")

options = ['int8', 'int32', 'float32']
precision = st.selectbox("select precision type: ", options)