import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def load_rf_assets():
    model = joblib.load("saved_models/rf_model.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")
    feature_names = joblib.load("data/feature_names.pkl")
    return model, X_test, y_test, feature_names


def compute_permutation_importance():
    model, X_test, y_test, feature_names = load_rf_assets()

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

    importances = result.importances_mean

    importance_df = list(zip(feature_names, importances))
    importance_df = sorted(importance_df, key=lambda x: x[1], reverse=True)

    return importance_df

def permutation_importance_plot():
    importance = compute_permutation_importance()
    features, values = zip(*importance[:10])
    
    fig, ax = plt.subplots()
    ax.barh(features, values)
    ax.invert_yaxis()
    ax.set_xlabel("Decreased in Accuracy")
    ax.set_title("Permutation Feature Importance (Random Forest)")
    st.pyplot(fig)