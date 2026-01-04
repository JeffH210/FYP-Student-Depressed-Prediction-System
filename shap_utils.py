import joblib
import streamlit as st

def fine_tuned_ann_predict_wrapper(X):
    from keras.models import load_model
    model = load_model("saved_models/fine_tune_ann_model.h5")
    return model.predict(X, verbose=0)

@st.cache_resource
def load_shap_explainer():
    data = joblib.load("modelsEvaluation/shap_explainer_fine_tuned_ann.pkl")
    return data["explainer"], data["feature_names"]