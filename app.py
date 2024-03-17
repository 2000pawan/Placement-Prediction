# Import Important Library.

import joblib
import streamlit as st 
from PIL import Image
import pandas as pd


# Load Model & Scaler & Polynomial Features

model=joblib.load('model.pkl')
sc=joblib.load('sc.pkl')

# Load DataSet
df=pd.read_csv('test.csv')

# Load Image

image=Image.open('img.jpg')

# Streamlit Function For Building Button & app.

def main():
    st.image(image,width=650)
    st.title('College Placement Prediction')
    html_temp='''
    <div style='background-color:red; padding:12px'>
    <h1 style='color:  #000000; text-align: center;'>College Placement Prediction Machine Learning Model</h1>
    </div>
    <h2 style='color:  red; text-align: center;'>Please Enter Input</h2>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    age=st.number_input('Enter Your Age.',value=None)
    gender= st.selectbox("Type or Select a Gender from the Dropdown(0->Female or 1->Male).",df['Gender'].unique()) 
    stream= st.selectbox("Type or Select a Stream from the Dropdown('Electronics And Communication':1 or 'Computer Science':2 or 'Information Technology':3 or 'Mechanical':4 or 'Electrical':5 or 'Civil':6).",df['Stream'].unique()) 
    internships= st.number_input('Enter Your Internships Count.',value=None)
    cgpa=st.number_input('Enter Your CGPA.',value=None)
    hostel= st.selectbox("Type or Select a Hostel from the Dropdown(0->DayScholar or 1->Hosteller).",df['Hostel'].unique()) 
    backlog= st.number_input('Enter Your Backlog.',value=None)
    input=[age,gender,stream,internships,cgpa,hostel,backlog]
    result=''
    if st.button('Predict',''):
        result=prediction(input)
    temp='''
     <div style='background-color:navy; padding:10px'>
     <h1 style='color: gold  ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp,unsafe_allow_html=True)
    

# Prediction Function to predict from model.

def prediction(input):
    test_input=sc.transform([input])
    predict=model.predict(test_input)
    if predict==0:
        return ('Your Placement Chance is very low (35-40%).Keep Updating yourself with new technology🥺🥺🥺🥺🥺.')
    else:
        return ('Your Placement Chance is good (85-95%). Keep Updating yourself with new technology😊😊😊😊😊.')

if __name__=='__main__':
    main()
