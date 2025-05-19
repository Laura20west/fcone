import streamlit as st
import pandas as pd
import numpy as np

# App title
st.title("Simple Streamlit App")

# Sidebar with user input
st.sidebar.header("User Input")
user_name = st.sidebar.text_input("Enter your name", "John Doe")
age = st.sidebar.slider("Select your age", 0, 100, 25)
color = st.sidebar.color_picker("Pick a color", "#00f900")

# Main content
st.write(f"Hello, {user_name}! You're {age} years old.")

# Display the selected color
st.write("Your selected color:")
st.markdown(f'<div style="width:100px; height:100px; background:{color};"></div>', 
            unsafe_allow_html=True)

# Data visualization section
st.header("Random Data Visualization")

# Generate random data
data = pd.DataFrame({
    'x': np.arange(1, 101),
    'y': np.random.randn(100).cumsum()
})

# Show the data
st.line_chart(data, x='x', y='y')

# Show raw data if user wants
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# Add a download button
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sample_data.csv',
    mime='text/csv',
)