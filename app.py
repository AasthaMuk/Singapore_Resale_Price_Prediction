# Importing the libraries
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime,date
import folium
from streamlit_folium import st_folium
import pickle
import requests
import statistics
import json
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import plotly.graph_objects as go

#---------------- Select the options -------------------------------------#
mrt_locations = pd.read_csv('mrt_loc.csv')
towns=['Select','ANG MO KIO','YISHUN','SENGKANG','HOUGANG']

streets=['Select','ANG MO KIO AVE 3' ,'YISHUN ST 31' ,'COMPASSVALE RD' ,'BUANGKOK CRES' ,'COMPASSVALE LANE']

flat_types =['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE',
            'MULTI GENERATION', 'MULTI-GENERATION']

block_no =['564' ,'333B','257C' ,'998A' ,'207D']

#------------------------------------------------------------------------#


#--------------------------- Functions ---------------------------------#

def set_page_config():
    icon = Image.open("images/icon.png")
    st.set_page_config(page_title= "Singapore Resale Predict",
                        page_icon= icon,
                        layout= "wide",
                        initial_sidebar_state= "expanded",
                        menu_items={'About': """# This dashboard app is created by Aastha Mukherjee!"""})
        
    st.markdown(""" 
            <style>
                    .stApp,[data-testid="stHeader"] {
                        background: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700014252.jpg");
                        background-size: cover
                    }

                    
                    .stSpinner,[data-testid="stMarkdownContainer"],.uploadedFile{
                       color:black !important;
                    }

                    [data-testid="stSidebar"]{
                       background: url("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77700014168.jpg");
                       background-size: cover
                    }

                    .stButton > button,.stDownloadButton > button {
                        background-color: #f54260;
                        color: black;
                    }

                    #custom-container {
                        background-color: #0B030F !important;
                        border-radius: 10px; /* Rounded corners */
                        margin: 20px; /* Margin */
                        padding: 20px;
                    }

            </style>""",unsafe_allow_html=True)
        

def style_submit_button():
    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #37a4de;
                                                        color: white!important;
                                                        width: 45%}
                    </style>
                """, unsafe_allow_html=True)
        
def home_page():
        left,right = st.columns((1,3))
        with right:
            st.markdown('<p style="color: black; font-size:45px; font-weight:bold">Singapore Resale Price Prediction</p>',unsafe_allow_html=True)
            st.markdown("""<p style="color: black; font-size:20px; font-weight:bold"> This application is mainly used to predict the HDB Flats managed by Singapore Government Agency and also do  some data analysis,exploration and vizualizations.</p>""",unsafe_allow_html=True)
            st.markdown('<br>',unsafe_allow_html=True)
            st.markdown("""<p style="color: black; font-size:18px; font-weight:bold">Click on the <span style="color: red; font-size:18px; font-weight:bold">Sidebar Menus</span> option to start exploring.</p>""",unsafe_allow_html=True)
            st.markdown('<p style="color: black; font-size:25px; font-weight:bold">TECHNOLOGIES USED :</p>',unsafe_allow_html=True)
            st.markdown("""
                            <p style="color: black; font-size:18px; font-weight:bold">*<span style="color: red; font-size:18px; font-weight:bold"> Python</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Streamlit</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Folium</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Matplotlib</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Seaborn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Scikit-Learn</span> *
                            <span style="color: red; font-size:18px; font-weight:bold"> Pickle</span></p>""",unsafe_allow_html=True)

def about_page():
            st.markdown("### :blue[Project Overview :]")
            st.markdown("<p style='color: black; font-size:20px; font-weight:bold'>This project aims to construct a machine learning model and implement "
                        "it as a user-friendly online application in order to provide accurate predictions about the "
                        "resale values of apartments in Singapore. </p>", unsafe_allow_html=True)
            st.markdown("<p style='color: black; font-size:20px; font-weight:bold'>This prediction model will be based on past transactions "
                        "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                        "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                        "of criteria, including location, the kind of apartment, the total square footage, and lease "
                        "remaining years. </p>",unsafe_allow_html=True)
            st.markdown("<p style='color: black; font-size:20px; font-weight:bold'>The provision of customers with an expected resale price based on these criteria is "
                        "one of the ways in which a predictive model may assist in the overcoming of these obstacles.</p>",unsafe_allow_html=True)
            
            st.markdown("### :blue[Domain :] Real Estate")



def geo_map():
        st.title("Singapore HDB flats using Folium")
        coor_df = pd.read_csv('df_flat_coordinates.csv')
        c1,c2 = st.columns((7,2))
        with c1:
            zoom_value = st.slider('Zoom Level',2,10,step=1)
            CONNECTICUT_CENTER = (1.375097,103.837619)
            map = folium.Map(location=CONNECTICUT_CENTER,zoom_start=zoom_value,no_wrap=True)
            i=0
            for index, row in coor_df.iterrows():
                if i==100:
                    break
                loc=[float(row['latitude']),float(row['longitude'])]
                folium.Marker(loc,tooltip=row['address']).add_to(map)
                i+=1
            st_folium(map,width=1500)



def bar_charts():
    df = pd.read_csv("final_data.csv")
    st.title("Some Bar Charts :")
    col1,col2,col3 = st.columns([1,0.2,1],gap="small")
    with col1:
        st.subheader("Flat Types present in Towns")
        fig = px.bar(df[0:7000],
                    title='Town Vs Flat Type',
                    x="town",
                    y="flat_type",
                    color="flat_type",
                    orientation='v',
                    color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)
    with col3:
        st.subheader("Resale Price based on Flat Type")
        fig = px.bar(df[0:7000],
                        title='Resale Price Vs Flat Type',
                        x="flat_type",
                        y="resale_price_log",
                        color="flat_type",
                        orientation='v',
                        color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(height=500,width=500)
        st.plotly_chart(fig,use_container_width=False)

    st.subheader("Count of HDB Flats by Town")
    fig = px.histogram(df, nbins=30, x="town", 
                       color="town", title="Flats Vs Town", 
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(
        xaxis_title='Towns',
        yaxis_title='No. Of HDB Flats'
    )
    fig.update_layout(height=500,width=800)
    st.plotly_chart(fig,use_container_width=False)

    block = df['block'].value_counts().index
    df_sorted = df.sort_values(by='block', ascending=False)
    top_100_descending = df_sorted.head(100)
    st.subheader("Count of HDB Flats for Resale based on Top 100 Blocks")
    fig = px.histogram(top_100_descending, nbins=len(block), x='block',
                       color="block", title="No. of Flats Vs Block", 
                       color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(
        xaxis_title='Block Name',
        yaxis_title='Count Of HDB Flats'
    )
    fig.update_layout(height=500,width=800)
    st.plotly_chart(fig,use_container_width=False)






def pie_charts():
     df = pd.read_csv("final_data.csv")
     col1,col2 = st.columns([1,1],gap="small")
     with col1:
        st.subheader("Floor Area Based on Flat Model")
        grouped_data = df.groupby('flat_model')['floor_area_sqm_log'].sum()
        fig = px.pie(grouped_data, 
                                    title='Floor Area vs Flat Model',
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.Agsunset,
                                    values='floor_area_sqm_log')
        fig.update_layout(height=650,width=500)
        st.plotly_chart(fig,use_container_width=True)
     with col2:
        st.subheader("Floor Area Based on Flat Type")
        grouped_data = df.groupby('flat_type')['floor_area_sqm_log'].sum()
        fig = px.pie(grouped_data, 
                                    title='Floor Area vs Flat Type',
                                    names=grouped_data.index,
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    values='floor_area_sqm_log')
        fig.update_layout(height=650,width=500)
        st.plotly_chart(fig,use_container_width=True)


     st.subheader("Resale Price Based on Town")
     fig = go.Figure(data=[go.Pie(labels=df['town'], values=df['resale_price_log'], hole=0.5)])
     fig.update_layout(title_text='Resale Price Vs Town')
     fig.update_layout(height=650,width=500)
     st.plotly_chart(fig)




def set_sidebar():
        with st.sidebar:
            selected = option_menu('Menu', ['Home Page',"Some Visualizations","Predict Value","About"],
                    icons=["house",'geo-fill','gear','flag','star'],
                    menu_icon= "menu-button-wide",
                    default_index=0,
                    styles={"nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "#6F36AD"},
                            "nav-link-selected": {"background-color": "#B1A3F7"}})

        if selected == 'Home Page':
            home_page()

        if selected == "About":
            about_page()


        if selected == 'Some Visualizations':
            geo_map()
            bar_charts()
            pie_charts()
        
        
        if selected == 'Predict Value':
            st.markdown('<p style="color: black; font-size:45px; font-weight:bold">Predicting the Resale Price</p>',unsafe_allow_html=True)
            
            with st.form("Predict_Resale_Price"):
                    col1,col2,col3 = st.columns([0.5,0.1,0.5])
                    # -----New Data inputs from the user for predicting the resale price-----
                    with col1:
                        select_town = st.selectbox(label='Town', options=towns)
                        select_street_name = st.selectbox(label='Street Name', options=streets)
                        select_flat_type = st.selectbox(label='Flat Type',options=flat_types)
                        block = st.selectbox(label='Block Number',options=block_no)
                    with col3:
                        floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
                        lease_commence_date = st.text_input(label='Lease Commence Date (Min: 1966 & Max: 2022)')
                        column1,column2,column3= st.columns([0.5,0.5,0.5])
                        column1.write("Select the Storey Range")
                        storey_min = column2.text_input('Min')
                        storey_max = column3.text_input('Max')
                    print(select_street_name)
                    # -----Submit Button for PREDICT RESALE PRICE-----
                    submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

                    if submit_button:
                        with st.spinner('Please wait, Work in Progress.....'):
                            time.sleep(2)
                            with open(r"regression_model.pkl", 'rb') as file:
                                print("loading regression model")
                                loaded_model = pickle.load(file)

                            with open(r'scaler.pkl', 'rb') as f:
                                print("loading scaler")
                                scaler_loaded = pickle.load(f)

                            # -----Calculating lease_remain_years using lease_commence_date-----
                            lease_remain_years = 99 - (2023 - int(lease_commence_date))

                            # -----Calculating median of storey_range to make our calculations quite comfortable-----
                            storey_median = statistics.median([int(storey_min),int(storey_max)])

                            # -----Getting the address by joining the block number and the street name-----
                            address = block + " " + select_street_name
                            query = 'https://www.onemap.gov.sg/api/common/elastic/search?searchVal='+str(address)+'&returnGeom=Y&getAddrDetails=Y'
                            resp = requests.get(query)

                            # -----Using OpenMap API getting the latitude and longitude location of that address-----
                            origin = []
                            data_geo_location = json.loads(resp.content)
                            if data_geo_location['found'] != 0:
                                latitude = data_geo_location['results'][0]['LATITUDE']
                                longitude = data_geo_location['results'][0]['LONGITUDE']
                                origin.append((latitude, longitude))

                            # -----Appending the Latitudes and Longitudes of the MRT Stations-----
                            list_of_mrt_coordinates = []
                            for lat, long in zip(mrt_locations['latitude'], mrt_locations['longitude']):
                                list_of_mrt_coordinates.append((lat, long))

                            # -----Getting distance to nearest MRT Stations (Mass Rapid Transit System)-----
                            list_of_dist_mrt = []
                            for destination in range(0, len(list_of_mrt_coordinates)):
                                list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                            shortest = (min(list_of_dist_mrt))
                            min_dist_mrt = shortest
                            list_of_dist_mrt.clear()

                            # -----Getting distance from CDB (Central Business District)-----
                            cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                            # -----Sending the user enter values for prediction to our model-----
                            new_sample = np.array(
                                [[np.log(floor_area_sqm), lease_remain_years,cbd_dist, min_dist_mrt,np.log(storey_median)]])
                            new_sample = scaler_loaded.transform(new_sample[:, :5])

                            new_pred = loaded_model.predict(new_sample)[0]
                            resale_price=str(np.exp(new_pred))
                            s="<p style='color: #8B120E; font-size:45px; font-weight:bold'>Predicted resale price: "+resale_price+"</p>"
                            st.balloons()
                            st.markdown(s,unsafe_allow_html=True)
                            
                        
                            # evalution metrics
                            with st.container():
                                actual = pd.DataFrame(index=[0],data=[307500])
                                predicted = pd.DataFrame(index=[0],data=[np.exp(new_pred)])
                                # Flattening the data
                                actual_values = actual.values.flatten()
                                predicted_values = predicted.values.flatten()
                                mse = mean_squared_error(actual_values,predicted_values)
                                # Normalization Method : to bring the value between 0 and 1
                                normalized_mse = mse / max(np.square(actual_values), np.square(predicted_values))
                                
                                mae = mean_absolute_error(actual,predicted)
                                # Normalization Method : to bring the value between 0 and 1
                                normalized_mae = mae / max(actual_values, predicted_values)

                                rmse = np.sqrt(normalized_mse)
                                st.write(" ")
                                st.markdown("<p style='color: black; font-size:30px; font-weight:bold'>Evaluation Metrics</p>",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Mean squared error:&emsp;"+str(float(normalized_mse))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Mean absolute error:&emsp;"+str(float(normalized_mae))+"</p",unsafe_allow_html=True)
                                st.markdown("<p style='color: #2A7DEF; font-size:25px; font-weight:bold'>Root mean squared error:&emsp;"+str(float(rmse))+"</p",unsafe_allow_html=True)
                        
                        

            
            
        
           
                    


#------------ Run the app --------------#
set_page_config()
set_sidebar()