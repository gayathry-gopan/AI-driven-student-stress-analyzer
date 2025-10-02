import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. Filepaths and Feature Definitions (MUST MATCH train_and_save_model.py) ---

# Define the file names for the saved assets
MODEL_FILE = 'stacking_model.pkl'
SCALER_FILE = 'scaler.pkl'
TARGET_ENCODER_FILE = 'target_encoder.pkl'
FEATURE_ENCODERS_FILE = 'feature_encoders.pkl'

# Define the final, confirmed feature lists used for model training
CATEGORICAL_FEATURES = ['Gender', 'Extracurricular_Activities', 'Parental_Education', 
                        'Financial_Support', 'Internet_Access_at_Home']
NUMERICAL_FEATURES = ['Age', 'Study_Time_Weekly', 'Attendance', 'Sleep_Hours', 
                      'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                      'Quizzes_Avg', 'Participation_Score', 'Projects_Score']
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_FEATURE = 'Stress_Level'

# --- 2. Asset Loading Function ---

@st.cache_resource
def load_assets():
    """Loads all necessary machine learning assets (model, scalers, encoders)."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        target_encoder = joblib.load(TARGET_ENCODER_FILE)
        feature_encoders = joblib.load(FEATURE_ENCODERS_FILE)
        
        # Use a consistent order of features expected by the model during prediction
        # The column order must be correct for the DataFrame passed to the model
        model_features_order = ALL_FEATURES
        
        return model, scaler, target_encoder, feature_encoders, model_features_order

    except FileNotFoundError as e:
        # This error handles the case where the user hasn't run the training script yet
        st.error(
            f"Error loading required file: {e.filename}. Please ensure you have run "
            f"'train_and_save_model.py' successfully to generate all .pkl files in the current directory."
        )
        # Return Nones so the app can render the error and stop further execution
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        return None, None, None, None, None

# Load all assets
model, scaler, target_encoder, feature_encoders, model_features_order = load_assets()

# Exit if assets failed to load
if model is None:
    st.stop()

# --- 3. Prediction Function ---

def predict_stress_level(input_data):
    """Preprocesses input data and makes a prediction."""
    
    # 1. Convert input dictionary to DataFrame, explicitly setting the correct order
    input_df = pd.DataFrame([input_data])[model_features_order]
    
    # --- Prepare Data Arrays ---
    
    # 2. Handle Categorical Features (Encoding)
    encoded_categorical_data = []
    for col in CATEGORICAL_FEATURES:
        encoder = feature_encoders[col]
        # Transform the single input value (as a list) and append the result
        # We ensure it's a string, then transform it to the numerical array
        encoded_value = encoder.transform(input_df[col].astype(str).tolist())
        encoded_categorical_data.append(encoded_value[0])
    
    # 3. Handle Numerical Features (Scaling)
    numerical_input = input_df[NUMERICAL_FEATURES].copy()
    # Scaler returns a NumPy array. We take the first (and only) row.
    scaled_numerical_features = scaler.transform(numerical_input)[0].tolist()
    
    # 4. Combine all features into one list, maintaining the training order (CAT then NUM)
    final_features_list = encoded_categorical_data + scaled_numerical_features
    
    # 5. Create the final, single-row DataFrame using the combined list and the exact feature names
    final_predict_df = pd.DataFrame(
        [final_features_list], 
        columns=ALL_FEATURES # Use the definitive list of ALL feature names
    )
    
    # 6. Make Prediction
    # CRITICAL FIX: Convert the DataFrame to a NumPy array using .values 
    # This bypasses the strict feature name check by the StackingClassifier's internal models.
    prediction_encoded = model.predict(final_predict_df.values)[0]
    
    # 7. Decode the prediction back to the original stress level (1-10)
    prediction_level = target_encoder.inverse_transform([prediction_encoded])[0]
    
    return prediction_level

# --- 4. Streamlit App Layout and Input ---

st.set_page_config(
    page_title="Student Stress Level Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Student Stress Level Predictor")
st.markdown("Use the sidebar to input student metrics and predict their stress level.")
st.divider()

# --- Sidebar Inputs ---

st.sidebar.header("Student Metrics Input")

# Dictionary to hold all user inputs
user_input = {}

# Age (Numerical)
user_input['Age'] = st.sidebar.slider("Age (Years)", 18, 30, 22)

# Study_Time_Weekly (Numerical)
user_input['Study_Time_Weekly'] = st.sidebar.slider("Study Hours Per Week", 1.0, 40.0, 15.0, step=0.5)

# Sleep_Hours (Numerical)
user_input['Sleep_Hours'] = st.sidebar.slider("Sleep Hours Per Night", 3.0, 10.0, 7.0, step=0.1)

# Attendance (%) (Numerical)
user_input['Attendance'] = st.sidebar.slider("Attendance (%)", 50.0, 100.0, 85.0, step=0.1)

# Gender (Categorical)
user_input['Gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Parental_Education (Categorical)
user_input['Parental_Education'] = st.sidebar.selectbox(
    "Parent's Education Level", 
    ["High School", "Bachelor's", "Master's", "PhD", "Associate's"]
)

# Financial_Support (Categorical)
user_input['Financial_Support'] = st.sidebar.selectbox(
    "Family Income Level", 
    ["Low", "Medium", "High"]
)

# Extracurricular_Activities (Categorical)
user_input['Extracurricular_Activities'] = st.sidebar.selectbox(
    "Extracurricular Activities", 
    ["Yes", "No"]
)

# Internet_Access_at_Home (Categorical)
user_input['Internet_Access_at_Home'] = st.sidebar.selectbox(
    "Internet Access at Home", 
    ["Yes", "No"]
)

st.sidebar.subheader("Score Metrics (0-100)")
# Numerical Score Inputs
user_input['Midterm_Score'] = st.sidebar.slider("Midterm Score", 0, 100, 75)
user_input['Final_Score'] = st.sidebar.slider("Final Score", 0, 100, 80)
user_input['Assignments_Avg'] = st.sidebar.slider("Assignments Average", 0, 100, 85)
user_input['Quizzes_Avg'] = st.sidebar.slider("Quizzes Average", 0, 100, 80)
user_input['Participation_Score'] = st.sidebar.slider("Participation Score", 0, 100, 90)
user_input['Projects_Score'] = st.sidebar.slider("Projects Score", 0, 100, 85)


# --- 5. Prediction Output ---

if st.button("Predict Stress Level", type="primary"):
    
    # Display the final prediction
    with st.spinner("Calculating prediction..."):
        try:
            # The input data is passed to the prediction function
            predicted_stress_raw = predict_stress_level(user_input)
            
            # Convert the predicted stress (which is a string '1'-'10') to an integer
            predicted_stress_int = int(predicted_stress_raw)
            
            # Determine Classification, Color, and Suggestion
            if predicted_stress_int <= 3:
                classification = "Low Stress"
                color = "green"
                suggestion = "Keep up the positive habits! Continue to balance studies with sufficient rest and social activities to maintain this level."
            elif predicted_stress_int <= 7:
                classification = "Medium Stress"
                color = "orange"
                suggestion = "Maintain your routine but integrate mindfulness or light physical activity. Schedule some dedicated relaxation time."
            else: # 8-10
                classification = "High Stress"
                color = "red"
                suggestion = "Seek immediate support. Focus on prioritizing essential tasks and taking short, frequent mental breaks."

            st.success("Prediction Complete!")

            st.markdown(
                f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #f0f2f6;
                    border-left: 5px solid {color};
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                ">
                    <p style="font-size: 1.2rem; margin: 0; color: #333;">Predicted Stress Classification:</p>
                    <h1 style="font-size: 3rem; color: {color}; margin: 5px 0 10px 0;">{classification}</h1>
                    <p style="font-size: 1.1rem; margin-top: 20px; font-weight: bold; color: #333;">Actionable Suggestion:</p>
                    <p style="font-size: 1rem; margin: 0; color: #555;">{suggestion}</p>
                    <p style="font-size: 0.8rem; margin-top: 15px; color: #888;">(Raw Score: {predicted_stress_raw} / 10)</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)

st.divider()
st.info("Input features and model logic are derived from the structure of your uploaded 'Students Performance Dataset.csv'.")
