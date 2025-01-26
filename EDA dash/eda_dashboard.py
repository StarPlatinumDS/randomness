#Lots of bugs but I ain't got time to fix em all

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load and display data
def load_data(file):
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1250']
    for encoding in encodings:
        try:
            data = pd.read_csv(file)
            return data
        except UnicodeDecodeError:
            st.error('Error: Unable to read file with standard encodings.')
            return None

#Basic EDA summary
def eda_sum(data):
    st.write('**Dataset Shape:**', data.shape)
    st.write('**Dataset Preview:**')
    st.dataframe(data.head())
    st.write('**Summary Statistics:**')
    st.write(data.describe())
    st.write('**Missing Values:**')
    st.write(data.isnull().sum())

#Correlation heatmap
def correlation_heatmap(data):
    st.write('### Correlation Heatmap')
    numeric_data = data.select_dtypes(include=['float64', 'int64']) #Numeric columns ONLY!
    if numeric_data.empty:
        st.warning('No numeric columns available.')
    else:
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot()

#Visualization
def visualize_data(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64']) #Numeric columns ONLY!
    if numeric_data.empty:
        st.warning('No numeric columns available for visualization.')
    else:

        st.write('### Histogram')
        column = st.selectbox('Select Column for Histogram', data.select_dtypes(include=['float64', 'int64']).columns)
        fig, ax = plt.subplots()
        data[column].plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='block')
        st.pyplot(fig)

        st.write('### Scatter Plot')
        x_axis = st.selectbox('Select X Axis', data.select_dtypes(include=['float64', 'int64']).columns)
        y_axis = st.selectbox('Select Y Axis', data.select_dtypes(include=['float64', 'int64']).columns)
        fig, ax = plt.subplots()
        data.plot(kind='scatter', x=x_axis, y=y_axis, ax=ax, color='green')
        st.pyplot(fig)

#Streamlit app
def main():
    st.title('📊 Basic EDA')
    st.write('Upload your CSV file to perform analysis')

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None: #Check for data's existence
            st.sidebar.header('Options')
            if st.sidebar.checkbox('Show Dataset Summary'):
                eda_sum(data)

            if st.sidebar.checkbox('Show Correlation Heatmap'):
                correlation_heatmap(data)

            if st.sidebar.checkbox('Visualize Data'):
                visualize_data(data)

if __name__ == '__main__':
    main()