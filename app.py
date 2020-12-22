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

# Settings
st.set_page_config(page_title = 'Foreseer', page_icon = ':dart:', layout = 'centered', initial_sidebar_state = 'auto')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Dataset
data = pd.read_csv('FinalData.csv')

# Load  model 
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
    
    city = st.sidebar.selectbox("Select city ", ("Coney Street, York City", 'Strøget, Copenhagen'))
    
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
data = pd.read_csv('FinalData.csv')


# About the prediction
st.sidebar.title('ML model')
st.sidebar.write('The prediction is based on a Random Forest Regressor with an 87.5% accuracy evaluated by R2. The Random Forest method is an ensemble method that combines several Decision Trees where the feature and threshold attribute of each node is chosen based on the Gini index:')
st.sidebar.latex('Gini = 1 -\sum_{k=0}^{K-1}{p_k}^2')
st.sidebar.write('Read more about the used Random Forest Regressor from scikit learn here:')
st.sidebar.markdown(
    """<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">Random Forest Regression</a>""", unsafe_allow_html=True,
)

# Team 
st.sidebar.title('Foreseer bio')
st.sidebar.write('We are a team of college students working on this project like it is our full time job. Any amount would help support and continue development on this project and is greatly appreciated.')
st.sidebar.write('_Foreseer, TechLabs 2020 &trade;_')

# Title
st.title('Welcome to Foreseer')
st.subheader('_Predict the crowd - select from the left sidebar_')

# Input
st.header('Your input ')
st.write('')
st.write(str(user_input_df['City'][0]), ':city_sunrise:', str(user_input_df['Full_date'][0]), ':date:', datetime.time(user_input_df['Full_time'][0]).strftime("%I:00 %p"), ':clock2:')

# Prediction
st.header('Our prediction ')

if user_input_df['City'][0] == 'Coney Street, York City':

    st.subheader(str(int(prediction)) + ' people' + ' :man-woman-girl-boy:')
    if prediction > data['TotalCount'].unique().mean():
        st.write('That is above the average of ' + str(int(data['TotalCount'].unique().mean())) + ' people in ' + str(user_input_df['City'][0] + ' :warning:'))
    else:
        st.write('That is below the average of ' + str(int(data['TotalCount'].unique().mean())) + ' people in ' + str(user_input_df['City'][0] + ' :white_check_mark:'))

    # Graph
    x = data['date_hour']
    y = data['TotalCount']

    plt.scatter(x, y,  cmap = 'inferno')
    plt.xlabel('Hour')
    plt.ylabel('Total Count')
    plt.plot(int(user_input_df['Full_time'][0]), int(prediction), 'ro', label = 'Prediction')
    plt.plot([0, 23], [int(data['TotalCount'].unique().mean()), int(data['TotalCount'].unique().mean())], label = 'Mean')

    t11 = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']

    plt.xticks(range(len(t11)), t11, size='small', rotation='vertical')

    plt.rcParams["figure.figsize"] = [12, 6]

    plt.legend()

    st.pyplot(plt)

    ############# About York City

    st.header('York City ')
    st.write('')

    image = Image.open('YorkCrowd.jpg')
    st.image(image, caption = 'Coney Street, York City', use_column_width = True)

    st.header('About')

    st.markdown('- Time zone: UTC+00:00')
    st.markdown('- Total area: 271.94 km2')
    st.markdown('- Total population: 210,618')

    st.subheader('Useful sources: ')
    st.write('- https://en.wikipedia.org/wiki/York')
    st.write('- https://www.york.gov.uk/')
    
elif user_input_df['City'][0] == 'Strøget, Copenhagen':
    
    st.subheader('Coming soon ' + ':exclamation:' + ':exclamation:')

############# About Copenhagen

    st.header('Copenhagen ')
    st.write('')

    image = Image.open('Copenhagen.jpg')
    st.image(image, caption = 'Strøget, Copenhagen', use_column_width = True)

    st.header('About')

    st.markdown('- Time zone: UTC+01:00')
    st.markdown('- Total area: 292.5 km2')
    st.markdown('- Total population: 794,128')

    st.subheader('Useful sources: ')
    st.write('- https://en.wikipedia.org/wiki/Copenhagen')
    st.write('- https://www.visitcopenhagen.com/')