import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
st.write("""
# Breast Cancer Prediction
Data obtained from the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
""")
st.sidebar.header('User Input Features')
uploaded_file=st.sidebar.file_uploader("Upload your input CSV file",type=["csv"])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else :
    def user_input_features():
        radius_mean=st.sidebar.slider('Mean radius (mm)',6.98,28.110,14.127)
        texture_mean=st.sidebar.slider('Mean Texture ',9.71,39.28,19.289)
        perimeter_mean=st.sidebar.slider('Mean Perimeter (mm)',43.79,188.5,91.969)
        area_mean=st.sidebar.slider('Mean Area (mm^2)',143.5,2501.00,654.889)
        smoothness_mean=st.sidebar.slider('Smoothness',0.0526,0.16340,0.096)
        compactness_mean=st.sidebar.slider('Compactness',0.019380,0.345,0.104)
        concavity_mean=st.sidebar.slider('Mean Concavity',0.0,0.4268,0.0887)
        concave_points_mean=st.sidebar.slider('Mean Concave points',0.0,0.201,0.088)
        symmetry_mean=st.sidebar.slider('Mean symmetry',0.106,0.304,0.1811)
        fractal_dimension_mean=st.sidebar.slider('Mean Fractal Dimension',0.04996,0.097440,0.06279)
        radius_se=st.sidebar.slider('Standard error in Radius',0.1115,2.873,0.405)
        texture_se=st.sidebar.slider('Standard error in texture',0.3602,4.885,1.21685)
        perimeter_se=st.sidebar.slider('Standard error in Perimeter',0.757 ,21.98,2.8660592267135288)
        area_se=st.sidebar.slider('Standard error in area',6.8020000000000005 ,542.2,40.33707908611603)
        smoothness_se=st.sidebar.slider('Standard error in smoothness',0.001713 ,0.03113,0.007040978910369071)
        compactness_se=st.sidebar.slider('Standard error in compactness',0.002252 ,0.1354,0.025478138840070306)
        concavity_se=st.sidebar.slider('Standard error in concavity',0.0 ,0.396,0.03189371634446394)
        concave_points_se=st.sidebar.slider('Standard error in concave points',0.0 ,0.05279,0.011796137082601056)
        symmetry_se=st.sidebar.slider('Standard error in concave points',0.007882,0.07895,0.020542298769771532)
        fractal_dimension_se=st.sidebar.slider('Standard error in fractal dimension',0.0008948000000000001,0.02984,0.0037949038664323374)
        radius_worst=st.sidebar.slider('worst radius',7.93 ,36.04,16.269189806678394)
        texture_worst=st.sidebar.slider('worst texture',12.02 ,49.54,25.677223198594014)
        perimeter_worst=st.sidebar.slider('worst perimeter',50.41 ,251.2,107.2612126537786)
        area_worst=st.sidebar.slider('worst area',185.2,4254.0,880.5831282952545)
        smoothness_worst=st.sidebar.slider('worst smoothness',0.07117000000000001,0.2226,0.13236859402460469)
        compactness_worst=st.sidebar.slider('worst compactness',0.02729 ,1.058,0.25426504393673144)
        concavity_worst=st.sidebar.slider('worst contactivity',0.0 ,1.252,0.27218848330404205)
        concave_points_worst=st.sidebar.slider('worst contactivity',0.0 ,0.29100000000000004,0.11460622319859404)
        symmetry_worst=st.sidebar.slider('worst symmetry',0.1565,0.6638,0.29007557117750454)
        fractal_dimension_worst=st.sidebar.slider('Fractal Dimesnion worst',0.05504,0.2075,0.08394581722319855)
        data={"radius_mean":radius_mean,
              "texture_mean":texture_mean,
              "perimeter_mean":perimeter_mean,
              "area_mean":area_mean,
              "smoothness_mean":smoothness_mean,
              "compactness_mean":compactness_mean,
               "concavity_mean":concavity_mean,
              "concave_points_mean":concave_points_mean,
              "symmetry_mean": symmetry_mean,
              "fractal_dimension_mean":fractal_dimension_mean,
              "radius_se":radius_se,
              "texture_se":texture_se,
              "perimeter_se":perimeter_se,
              "area_se":area_se,
              "smoothness_se":smoothness_se,
              "compactness_se":compactness_se,
              "concavity_se":concavity_se,
              "concave_points_se":concave_points_se,
              "symmetry_se":symmetry_se,
              "fractal_dimension_se":fractal_dimension_se,
              "radius_worst":radius_worst,
              "texture_worst":texture_worst,
              "perimeter_worst":perimeter_worst,
              "area_worst":area_worst,
              "smoothness_worst":smoothness_worst,
              "compactness_worst":compactness_worst,
              "concavity_worst":concavity_worst,
              "concave_points_worst":concave_points_worst,
              "symmetry_worst":symmetry_worst,
              "fractal_dimension_worst":fractal_dimension_worst}
        features=pd.DataFrame(data,index=[0])
        return features
    input_df=user_input_features()
load_clf=pickle.load(open('cancer_voting_clf.pkl','rb'))
prediction=load_clf.predict(input_df)
st.subheader('Prediction')
if prediction ==0:
    st.header("Benign")
else :
    st.header("Malignant")
