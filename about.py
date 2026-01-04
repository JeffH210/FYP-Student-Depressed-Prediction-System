import streamlit as st

def about_info():
    st.title("About the Application")
    with st.expander("View Details"):
        st.markdown("""
        ### DL vs ML Models in Student Depression Prediction

        This app demonstrates the comparison of **Deep Learning (DL)** and **Machine Learning (ML)** models for predicting student depression based on Kaggle dataset feature.
        Student Depression Prediction Using Fine Tune ANN model

        **Key Features:**
        - Compare model performance using **accuracy**, **ROC curves**.
        - Deep Learning models (e.g., ANN, Fine-Tuned ANN) versus traditional Machine Learning models.
        - Helps visualize how different models perform on the same dataset.

        **Objective:**  
        To provide insights into which type of model is more effective for predicting depression in students, and to help understand model evaluation metrics.

        **References:**  
        - Python, TensorFlow/Keras, scikit-learn, joblib, matplotlib, seaborn, sklearn, numpy  
        - Streamlit for interactive dashboard visualization
        """)