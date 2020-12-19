import numpy as np 
import pandas as pd 
import joblib
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image
import datetime
from datetime import date

st.set_page_config(page_title = 'Foreseer', page_icon = ':dart:', layout = 'centered', initial_sidebar_state = 'auto')

data = pd.read_csv('FinalData.csv')

# Load  model a 
model = pickle.load(open('model.pkl', 'rb'))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    
    int_features = user_input_df.values[0]

    int_features[0] = int_features[0] - 1
    
    int_features[1] = int_features[1] - 1

    dummy_features = []

    month = list(range(12))
    day_of_month = list(range(31))
    hour = list(range(24))
    day_of_week = list(range(7))

    # Get dummy data
    dummy_month = pd.get_dummies(month)
    dummy_day_of_month = pd.get_dummies(day_of_month)
    dummy_hour = pd.get_dummies(hour)
    dummy_day_of_week = pd.get_dummies(day_of_week)

    # Print dummy data for input
    dummy_features.extend(dummy_month[int_features[0]])
    dummy_features.extend(dummy_day_of_month[int_features[1]])
    dummy_features.extend(dummy_day_of_week[int_features[2]])
    dummy_features.extend(dummy_hour[int_features[3]])

    # # final array
    final = np.array(dummy_features)

    return final


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    st.sidebar.title('Select here ' + ':point_down:')
    
    city = st.sidebar.selectbox("Select city ", ("York City", ))
    
    date_input = st.sidebar.date_input('Select date', min_value = date.today())
    
    time_input = st.sidebar.slider('Select time', 0, 23, 1)
    
    input_month = date_input.month
    input_date_of_month = date_input.day
    input_day_of_week = date_input.weekday()
    input_hour = time_input
    
    features = {
            'Month': input_month,
            'Date_of_month': input_date_of_month,
            'Day_of_week': input_day_of_week,
            'Hour': input_hour,
            'Full_date': date_input,
            'Full_time': time_input, 
            'City': city}
    
    data = pd.DataFrame(features, index=[0])

    return data

user_input_df = get_user_input()

processed_user_input = data_preprocessor(user_input_df)

prediction = model.predict([processed_user_input])


#####################################################################################
# Frontend


st.title('Welcome to Foreseer')
st.subheader('Predict the crowd - Choose your city from the left sidebar')

st.header('Your input ')
st.write('')
st.write(str(user_input_df['City'][0]), ':city_sunrise:', str(user_input_df['Full_date'][0]), ':date:', datetime.time(user_input_df['Full_time'][0]).strftime("%I:00 %p"), ':clock2:')

############# Prediction

st.header('Our prediction ')
st.subheader(':point_right: ' + str(int(prediction)) + ' people' + ' :man-woman-girl-boy:')

# Graph

# The model

############# About York City

st.header('York City ')
st.write('')

image = Image.open('YorkCity.jpg')
st.image(image, caption = 'York City', use_column_width = True)

st.header('About')

st.markdown('- Average crowd: 923')
st.markdown('- Total area: 271.94 km2')
st.markdown('- Total population: 210,618')

st.subheader('Useful sources: ')
st.write('- https://en.wikipedia.org/wiki/York')
st.write('- https://www.york.gov.uk/')