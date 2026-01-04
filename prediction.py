import joblib
import numpy as np
import shap
from keras.models import load_model
from shap_utils import load_shap_explainer

# Load Fine Tune ANN model
fine_tune_ann_model = load_model("saved_models/fine_tune_ann_model.h5")

le_gender = joblib.load("preprocessingObjects/le_gender.pkl")
le_suicidal = joblib.load("preprocessingObjects/le_suicidal.pkl")
le_family = joblib.load("preprocessingObjects/le_family.pkl")
scaler = joblib.load("preprocessingObjects/scaler.pkl")

sleep_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
    "Others": 4
}

dietary_habits_map = {
    'Healthy':0,
    'Moderate':1,
    'Unhealthy':2,
    'Others': 3
}


def predict_depression(
        gender, age, academic_pressure, cgpa, study_satisfaction,
        sleep_duration, dietary_habits, suicidal_thoughts, 
        study_hours, financial_stress, family_history
):
    gender_encoded = le_gender.transform([gender])[0]
    suicidal_encoded = le_suicidal.transform([suicidal_thoughts])[0]
    family_encoded = le_family.transform([family_history])[0]
    sleep_encoded = sleep_map[sleep_duration]
    dietary_habits_encoded = dietary_habits_map[dietary_habits]

    numeric_features = np.array([[age, academic_pressure, cgpa, study_satisfaction, study_hours, financial_stress]])
    numeric_scaled = scaler.transform(numeric_features)

    categorical_features = np.array([[gender_encoded, sleep_encoded, dietary_habits_encoded, suicidal_encoded, family_encoded]])
    X_input_scaled = np.hstack([numeric_scaled, categorical_features])

    raw_inputs = {
        "Age": age,
        "Academic Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study Sastisfaction": study_satisfaction,
        "Study Hours": study_hours,
        "Financial Stress": financial_stress,
        "Gender": gender,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Suicidal Thoughts": suicidal_thoughts,
        "Family History of Mental Illness": family_history
    }

    probability = fine_tune_ann_model.predict(X_input_scaled, verbose=0)[0][0]
    prediction = "Depressed" if probability > 0.5 else "Not Depressed"
    return prediction, probability, X_input_scaled, raw_inputs

def explain_prediction(X_input_scaled):
    explainer, feature_names = load_shap_explainer()

    # shap_values could be a list for binary classification
    shap_values = explainer.shap_values(X_input_scaled)
    if isinstance(shap_values, list):  # binary classification
        shap_values = np.array(shap_values[1])  # class 1 (Depressed)
    else:
        shap_values = np.array(shap_values)

    if shap_values.ndim == 1:  # single sample
        shap_values = shap_values.reshape(1, -1)

    contributions = list(zip(feature_names, shap_values[0]))
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    # expected_value is a float for class 1
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1])  # class 1
    else:
        expected_value = float(expected_value)

    return contributions, shap_values, feature_names, expected_value

def show_risk(probability):
    if probability < 0.4:
        risk = "Low Risk"
    
    elif probability < 0.7:
        risk = "Moderate Risk"
    
    else:
        risk = "High Risk"

    return risk