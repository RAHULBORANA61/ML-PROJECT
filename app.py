%%writefile app.py
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("/content/drive/My Drive/ML LAB/rb.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('/content/drive/My Drive/ML LAB/diabetes2.csv')
X = dataset.iloc[:, 0:3].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Glucose,BloodPressure,Age):
  output= model.predict(sc.transform([[Glucose,BloodPressure,Age]]))
  print("Diabetic", output)
  if output==[1]:
    prediction="Person is Diabetic"
  else:
    prediction="Person is not Diabetic"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Group of Institute</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>  
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Diabet Prediction")
    Glucose = st.number_input(" ")
    BloodPressure = st.number_input(" ")
    Age = st.number_input(" ")
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Glucose,BloodPressure,Age)
      st.success('Model has predicted {}'.format(result))

if __name__=='__main__':
  main()
   
