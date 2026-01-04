import streamlit as st
import matplotlib.pyplot as plt
import shap
from shap_utils import load_shap_explainer
from prediction import explain_prediction

def shap_plot_waterfall(shap_values, X_input_scaled, feature_names, expected_value):
    """
    shap_values: np.array of shape (1, n_features)
    X_input_scaled: np.array of shape (1, n_features)
    feature_names: list of feature names
    expected_value: float (base value)
    """
    expl = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=X_input_scaled[0],
        feature_names=feature_names
    )
    shap.plots.waterfall(expl, show=False)
    st.pyplot(plt.gcf())
    plt.clf()