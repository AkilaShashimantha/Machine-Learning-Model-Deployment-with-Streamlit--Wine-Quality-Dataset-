import streamlit as st
import pandas as pd
import pickle

# --- Load Model and Scaler ---
try:
    with open('notebooks/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('notebooks/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the training notebook first.")
    st.stop()

# --- Load Dataset for Context ---
try:
    wine_df = pd.read_csv('notebooks/WineQT.csv').drop('Id', axis=1)
except FileNotFoundError:
    st.error("WineQT.csv not found. Make sure it's in the root directory.")
    st.stop()


# --- Streamlit App ---
st.title('🍷 Wine Quality Prediction')
st.write("""
This app predicts whether a wine is of **good** or **bad** quality based on its chemical properties.
[cite_start]Input the wine's features using the sliders in the sidebar. [cite: 32, 43]
""")

# --- Sidebar for User Input ---
st.sidebar.header('Input Wine Features')

def user_input_features():
    """Creates sliders in the sidebar for user input."""
    fixed_acidity = st.sidebar.slider('Fixed Acidity', float(wine_df['fixed acidity'].min()), float(wine_df['fixed acidity'].max()), float(wine_df['fixed acidity'].mean()))
    volatile_acidity = st.sidebar.slider('Volatile Acidity', float(wine_df['volatile acidity'].min()), float(wine_df['volatile acidity'].max()), float(wine_df['volatile acidity'].mean()))
    citric_acid = st.sidebar.slider('Citric Acid', float(wine_df['citric acid'].min()), float(wine_df['citric acid'].max()), float(wine_df['citric acid'].mean()))
    residual_sugar = st.sidebar.slider('Residual Sugar', float(wine_df['residual sugar'].min()), float(wine_df['residual sugar'].max()), float(wine_df['residual sugar'].mean()))
    chlorides = st.sidebar.slider('Chlorides', float(wine_df['chlorides'].min()), float(wine_df['chlorides'].max()), float(wine_df['chlorides'].mean()))
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', float(wine_df['free sulfur dioxide'].min()), float(wine_df['free sulfur dioxide'].max()), float(wine_df['free sulfur dioxide'].mean()))
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', float(wine_df['total sulfur dioxide'].min()), float(wine_df['total sulfur dioxide'].max()), float(wine_df['total sulfur dioxide'].mean()))
    density = st.sidebar.slider('Density', float(wine_df['density'].min()), float(wine_df['density'].max()), float(wine_df['density'].mean()), format="%.4f")
    ph = st.sidebar.slider('pH', float(wine_df['pH'].min()), float(wine_df['pH'].max()), float(wine_df['pH'].mean()))
    sulphates = st.sidebar.slider('Sulphates', float(wine_df['sulphates'].min()), float(wine_df['sulphates'].max()), float(wine_df['sulphates'].mean()))
    alcohol = st.sidebar.slider('Alcohol', float(wine_df['alcohol'].min()), float(wine_df['alcohol'].max()), float(wine_df['alcohol'].mean()))

    data = {
        'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity, 'citric acid': citric_acid,
        'residual sugar': residual_sugar, 'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide, 'density': density, 'pH': ph, 'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display User Input ---
st.subheader('Your Input Parameters')
st.write(input_df)

# --- Prediction ---
if st.button('Predict Quality'):
    with st.spinner('Predicting...'):
        # Scale the user input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader('Prediction Result')
        quality = 'Good' if prediction[0] == 'good' else 'Bad'
        st.success(f'The predicted quality of the wine is: **{quality}**')

        st.subheader('Prediction Probability')
        st.write(f"Probability of being 'good' quality: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of being 'bad' quality: {prediction_proba[0][0]:.2f}")