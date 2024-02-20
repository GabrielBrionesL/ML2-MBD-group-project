import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/train.csv")

    # Convert categorical columns to 'category' type
    data['dental caries'] = data['dental caries'].astype('category')
    data['smoking'] = data['smoking'].astype('category')
    data['hearing(left)'] = data['hearing(left)'].astype('category')
    data['hearing(right)'] = data['hearing(right)'].astype('category')
    data['Urine protein'] = data['Urine protein'].astype('category')

    return data

def add_sidebar():
    st.sidebar.header("Bio Signals")
    
    data = get_clean_data()

    # Define the labels
    slider_labels = [
    ("Height (cm)", "height(cm)"),
    ("Hemoglobin", "hemoglobin"),
    ("GTP", "Gtp"),
    ("Triglyceride", "triglyceride"),
    ("Weight (kg)", "weight(kg)"),
    ("Serum Creatinine", "serum creatinine"),
    ("Age", "age"),
    ("HDL", "HDL"),
    ("ALT", "ALT"),
    ("LDL", "LDL"),
]

    input_dict = {}

    # Add the sliders
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['smoking'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict
    
def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Height (cm)', 'Hemoglobin', 'GTP', 'Triglyceride', 
                'Weight (kg)', 'Serum Creatinine', 
                'Age', 'HDL',
                'ALT', 'LDL']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
    r=[
        input_data['height(cm)'], input_data['hemoglobin'], input_data['Gtp'],
        input_data['triglyceride'], input_data['weight(kg)'], input_data['serum creatinine'],
        input_data['age'], input_data['HDL'], input_data['ALT'], input_data['LDL']
    ],
    theta=categories,
    fill='toself',
    name='Mean Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
    input_array = np.array(list(input_data.values())).reshape(1, -1)
  
    input_array_scaled = scaler.transform(input_array)
  
    prediction = model.predict(input_array_scaled)
  
    st.subheader("Smoking or Non-Smoking Prediction")
    st.write("The person is classified as:")
  
    if prediction[0] == 0:
        st.write("<span class='diagnosis nonsmoker'>Non-Smoker</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis smoker'>Smoker</span>", unsafe_allow_html=True)
    
  
    st.write("Probability of being a non-smoker: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being a smoker: ", model.predict_proba(input_array_scaled)[0][1])
  
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="Binary Prediction of Smoker Status using Bio-Signals",
        page_icon=":smoking:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Binary Prediction of Smoker Status using Bio-Signals")
        st.write("Please connect this app to your clinical lab to help diagnose smoking status from your patient information. Using an XGBoost machine learning model, this app predicts whether an individual is a smoker or non-smoker based on the bio-signals it receives from your clinical lab. You can also update the measurements by hand using the sliders in the sidebar.")
    
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()