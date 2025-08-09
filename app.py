import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# --- Load Model and Scaler ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the training notebook first.")
    st.stop()

# --- Load Dataset for Context ---
try:
    wine_df = pd.read_csv('data/WineQT.csv').drop('Id', axis=1)
except FileNotFoundError:
    st.error("WineQT.csv not found. Make sure it's in the root directory.")
    st.stop()


# --- Streamlit App ---
st.title('🍷 Wine Quality Prediction')
st.write("""
This app predicts whether a wine is of **good** or **bad** quality based on its chemical properties.
Input the wine's features using the sliders in the sidebar. 
""")

# --- Sidebar for User Input ---
st.sidebar.header('Input Wine Features')

def user_input_features():
    """Creates sliders in the sidebar for user input."""
    # Store default values
    defaults = {
        'fixed acidity': float(wine_df['fixed acidity'].mean()),
        'volatile acidity': float(wine_df['volatile acidity'].mean()),
        'citric acid': float(wine_df['citric acid'].mean()),
        'residual sugar': float(wine_df['residual sugar'].mean()),
        'chlorides': float(wine_df['chlorides'].mean()),
        'free sulfur dioxide': float(wine_df['free sulfur dioxide'].mean()),
        'total sulfur dioxide': float(wine_df['total sulfur dioxide'].mean()),
        'density': float(wine_df['density'].mean()),
        'pH': float(wine_df['pH'].mean()),
        'sulphates': float(wine_df['sulphates'].mean()),
        'alcohol': float(wine_df['alcohol'].mean())
    }
    
    fixed_acidity = st.sidebar.slider('Fixed Acidity', float(wine_df['fixed acidity'].min()), float(wine_df['fixed acidity'].max()), defaults['fixed acidity'])
    volatile_acidity = st.sidebar.slider('Volatile Acidity', float(wine_df['volatile acidity'].min()), float(wine_df['volatile acidity'].max()), defaults['volatile acidity'])
    citric_acid = st.sidebar.slider('Citric Acid', float(wine_df['citric acid'].min()), float(wine_df['citric acid'].max()), defaults['citric acid'])
    residual_sugar = st.sidebar.slider('Residual Sugar', float(wine_df['residual sugar'].min()), float(wine_df['residual sugar'].max()), defaults['residual sugar'])
    chlorides = st.sidebar.slider('Chlorides', float(wine_df['chlorides'].min()), float(wine_df['chlorides'].max()), defaults['chlorides'])
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', float(wine_df['free sulfur dioxide'].min()), float(wine_df['free sulfur dioxide'].max()), defaults['free sulfur dioxide'])
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', float(wine_df['total sulfur dioxide'].min()), float(wine_df['total sulfur dioxide'].max()), defaults['total sulfur dioxide'])
    density = st.sidebar.slider('Density', float(wine_df['density'].min()), float(wine_df['density'].max()), defaults['density'], format="%.4f")
    ph = st.sidebar.slider('pH', float(wine_df['pH'].min()), float(wine_df['pH'].max()), defaults['pH'])
    sulphates = st.sidebar.slider('Sulphates', float(wine_df['sulphates'].min()), float(wine_df['sulphates'].max()), defaults['sulphates'])
    alcohol = st.sidebar.slider('Alcohol', float(wine_df['alcohol'].min()), float(wine_df['alcohol'].max()), defaults['alcohol'])

    data = {
        'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity, 'citric acid': citric_acid,
        'residual sugar': residual_sugar, 'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide, 'density': density, 'pH': ph, 'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(data, index=[0])
    
    # Track which features have been changed from defaults
    changed_features = []
    for feature, value in data.items():
        if abs(value - defaults[feature]) > 0.001:  # Small tolerance for floating point comparison
            changed_features.append(feature)
    
    return features, changed_features, defaults

input_df, changed_features, defaults = user_input_features()

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
        
        if quality == 'Bad':
            st.markdown(
                f"""
                <div style="
                    background-color: #ff4444;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 18px;
                    font-weight: bold;
                    transition: all 0.3s ease;
                    cursor: pointer;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 8px 16px rgba(0,0,0,0.3)';" 
                   onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';">
                    🍷 The predicted quality of the wine is: <strong>{quality}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.success(f'🍷 The predicted quality of the wine is: **{quality}**')

        st.subheader('Prediction Probability')
        st.write(f"Probability of being 'good' quality: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of being 'bad' quality: {prediction_proba[0][0]:.2f}")
        
        # --- Feature Impact Visualization ---
        st.subheader('🔍 Why This Prediction?')
        
        if not changed_features:
            st.info("💡 **Using default values**: Adjust the sliders in the sidebar to see how different wine characteristics impact the prediction!")
        else:
            st.write("This chart shows how your **changed** wine features impact the prediction:")
            
            # Color guide outside the chart
            color_name = 'Red' if quality == 'Bad' else 'Green'
            st.markdown(f"""
            **📊 Chart Color Guide:**
            - **Darker {color_name}**: Feature value above average (↑)
            - **Lighter {color_name}**: Feature value below average (↓)
            """)
            
            # Get feature names and user values - only for changed features
            feature_names = [f for f in input_df.columns if f in changed_features]
            user_values = [input_df[f].iloc[0] for f in feature_names]
            
            # Calculate feature impact based on deviation from mean and feature importance
            feature_impacts = []
            feature_deviations = []
            
            for i, feature in enumerate(feature_names):
                # Calculate how much user's value deviates from dataset mean
                feature_mean = wine_df[feature].mean()
                feature_std = wine_df[feature].std()
                
                # Normalize the deviation
                deviation = (user_values[i] - feature_mean) / feature_std if feature_std > 0 else 0
                
                # Get feature index in original dataframe to get importance
                feature_idx = list(input_df.columns).index(feature)
                # Calculate impact: deviation * feature importance from model
                impact = abs(deviation) * model.feature_importances_[feature_idx]
                
                feature_impacts.append(impact)
                feature_deviations.append(deviation)
            
            # Create a DataFrame for easier handling
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': feature_impacts,
                'User_Value': user_values,
                'Deviation': feature_deviations
            })
            
            # Sort by impact (ascending for horizontal bar chart)
            impact_df = impact_df.sort_values('Impact', ascending=True)
            
            # Create the bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color bars based on deviation direction and prediction quality
            bar_colors = []
            for deviation in impact_df['Deviation']:
                if quality == 'Bad':
                    # Red gradient for bad predictions
                    if deviation > 0:
                        bar_colors.append('#ff4444')  # Bright red for positive deviations
                    else:
                        bar_colors.append('#ff8888')  # Lighter red for negative deviations
                else:
                    # Green gradient for good predictions
                    if deviation > 0:
                        bar_colors.append('#44aa44')  # Bright green for positive deviations
                    else:
                        bar_colors.append('#88cc88')  # Lighter green for negative deviations
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(impact_df)), impact_df['Impact'], color=bar_colors)
            
            # Customize the chart
            ax.set_yticks(range(len(impact_df)))
            ax.set_yticklabels([f"{feat}\n(Value: {val:.2f})" for feat, val in 
                               zip(impact_df['Feature'], impact_df['User_Value'])])
            ax.set_xlabel('Feature Impact on Prediction (Higher = More Influential)')
            ax.set_title(f'Changed Features Impact on {quality} Quality Prediction', 
                        fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (impact, deviation) in enumerate(zip(impact_df['Impact'], impact_df['Deviation'])):
                direction = "↑" if deviation > 0 else "↓" if deviation < 0 else "="
                ax.text(impact + 0.001, i, f'{impact:.3f} {direction}', 
                       va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # --- Feature Explanations ---
        if changed_features:
            st.subheader('📚 Feature Explanations')
            
            # Get top 3 most impactful features
            top_features = impact_df.tail(3)
        
            feature_explanations = {
                'fixed acidity': 'Amount of non-volatile acids (tartaric acid). Higher values can make wine taste sharp.',
                'volatile acidity': 'Amount of volatile acids (acetic acid). High levels can cause vinegar taste.',
                'citric acid': 'Adds freshness and flavor to wine. Found in small quantities.',
                'residual sugar': 'Amount of sugar left after fermentation. Affects sweetness.',
                'chlorides': 'Amount of salt in wine. High levels can affect taste.',
                'free sulfur dioxide': 'Prevents microbial growth and wine oxidation.',
                'total sulfur dioxide': 'Total amount of SO2. Too much can be detected in taste and smell.',
                'density': 'Density of wine. Related to alcohol and sugar content.',
                'pH': 'Acidity level. Lower pH = more acidic.',
                'sulphates': 'Wine additive that contributes to SO2 levels.',
                'alcohol': 'Alcohol percentage. Affects wine body and taste.'
            }
            
            st.write("**Top 3 most impactful changed features:**")
            for _, row in top_features.iterrows():
                feature = row['Feature']
                impact = row['Impact']
                value = row['User_Value']
                
                # Get wine dataset statistics for comparison
                feature_mean = wine_df[feature].mean()
                feature_std = wine_df[feature].std()
                
                # Determine if value is high, normal, or low
                if value > feature_mean + feature_std:
                    level = "**High**"
                    color = "🔴"
                elif value < feature_mean - feature_std:
                    level = "**Low**"
                    color = "🔵"
                else:
                    level = "**Normal**"
                    color = "🟢"
                
                st.write(f"{color} **{feature.title()}**: {level} (Your value: {value:.2f})")
                st.write(f"   {feature_explanations.get(feature, 'No explanation available.')}")
                st.write(f"   Impact on your prediction: {impact:.3f}")
                st.write("---")