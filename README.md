# 🍷 Wine Quality Prediction with Streamlit

This project is an assignment to build a complete machine learning pipeline, from data exploration to model deployment. [cite: 4] [cite_start]It uses the Wine Quality dataset to train a classification model and deploys it as an interactive web application using Streamlit and Streamlit Cloud. 

## Dataset

The dataset used for this project is the **Wine Quality Dataset** from Kaggle (`WineQT.csv`). It contains 11 chemical properties of wine (e.g., fixed acidity, alcohol, pH) and a quality score. For this project, the quality score was converted into a binary target: 'good' (score >= 6) and 'bad' (score < 6).

## Model Training

Two different algorithms were trained to predict wine quality: 
* Logistic Regression
* Random Forest Classifier

The best performing model was the **Random Forest Classifier**, which was selected for deployment. [cite_start]The model training and evaluation process is detailed in the `model_training.ipynb` notebook. 

## How to Run the Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repository URL]
    cd [Your-Project-Directory-Name]
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Deployed Application

You can access the deployed application here: https://her6bf4jzzdhx8ecmxurvd.streamlit.app/
