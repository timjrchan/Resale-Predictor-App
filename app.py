# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Resale Unit', page_icon=':office:', layout='wide', initial_sidebar_state='collapsed')

# Set title of the app
st.title(':office: Predict Valuation of the Unit')

# Load data
#df = pd.read_csv('iris.csv')
hdb = pd.read_csv('hdb_cleaned2.csv'
                 , index_col=0, low_memory=False)


left, right = st.columns((4,4))

# Set input widgets
#st.sidebar.subheader('Select the following attributes')
num_rooms = left.selectbox('No. of Rooms', ('One Room', 'Two Room', 'Three Room', 'Four Room', 'Five Room', 'Executive',
                                 'Multigeneration'))

floor_cat = right.selectbox('floor cat', (3, 2, 1))

floor_area = right.slider('What is the area of the house (sqm)',70, 200)

age_hdb = left.slider('The age of the HDB', 5, 70)
max_floor = right.slider('What is the maximum floor?', 2, 50)
mrt_distance = left.slider('Distance to nearest MRT', 0, 2500)


# Input default values


# Separate to X and y
y = hdb['resale_price']
X = hdb.drop(columns = ['resale_price', 'floor_type', 'pri_sch_name', 'mrt_name', 'planning_area','id', 'flat_type',
                        'nearby_top_sch','prop_one_room','prop_two_room', 'prop_three_room', 'prop_four_room', 
                        'prop_five_room', 'prop_exec', 'prop_multigen', 'prop_studio_apt',
                        'flat_model_Adjoined flat', 'flat_model_Apartment', 'flat_model_DBSS', 
                          'flat_model_Improved', 'flat_model_Improved-Maisonette', 'flat_model_Maisonette',
                            'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2',
                              'flat_model_Multi Generation', 'flat_model_New Generation', 
                              'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft', 
                              'flat_model_Premium Maisonette', 'flat_model_Simplified', 
                              'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1', 
                              'flat_model_Type S2',
                              'region_East', 'region_North', 'region_North East','region_West', 
                              'North South Line', 'North East line', 'East West Line', 'Circle Line', 
                              'Down Town Line', 'Thomson East Coast Line'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Build model
model = LinearRegression()
model.fit(X_train, y_train)



# set default values
tranc_year_default = 2024
tranc_month_default = 3
multistorey_cp_default = 1
mall_within_2km_default = 1
hawker_within_2km_default = 1
bus_interchange_default = 0
floor_density_default = 0.5





# Generate prediction based on user selected attributes
y_pred = model.predict([[floor_area, tranc_year_default, tranc_month_default, 
                        age_hdb, max_floor, multistorey_cp_default, mall_within_2km_default, 
                        hawker_within_2km_default, mrt_distance, bus_interchange_default,  floor_density_default, floor_cat]])

string_to_num_dict = {}

add_on_constant = {'One Room':0.8, 'Two Room':1.4, 'Three Room':2.8, 'Four Room':3.7, 'Five Room':4.3, 'Executive':4.7,
                                 'Multigeneration':5.4}

estimated_value = round(y_pred[0] + 88888*add_on_constant[f'{num_rooms}'],-3)


#r2_val = metrics.r2_score(y_test, y_pred)

# Display EDA
st.header('Estimated Valuation:')
#st.write('R2 value of this model')
#groupby_species_mean = hdb.groupby('Species').mean()
st.markdown(f'<h1 style="text-align: left;">${estimated_value:.0f}</h1>', unsafe_allow_html=True)

# Disclaimer
st.write('---')
st.write('**Disclaimer:**')
st.write('Please be aware that this model is still under development, and the valuations provided are for demonstration purposes only.')





