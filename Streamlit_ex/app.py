import streamlit as st
import pandas as pd
import numpy as np

st.title("Machine Learning Application")

st.write("Hi, Welcome to the machine learning world")

dataframe = pd.DataFrame({
    "first_column": [1,2,3,4],
    "second_column": [7,8,9,10]
})

st.write("here is the simple dataframe")
st.write(dataframe)

##Line chart

chart_data = pd.DataFrame(
    np.random.randn(20, 3), columns=['a', 'b', 'c']
)

st.line_chart(chart_data)