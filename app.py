import streamlit as st
import pandas as pd
from home import home_page
from sliderbar import navigation_slidebar
from preprocessing import preprocess_pipeline
from eda import histogram_plot, correlation_heatmap_plot, boxplot
from factor_risk import factor_risks_analysis
from modelComparison import plot_train_test_accuracy
from roc_auc_plot import roc_auc_curve_plot
from prediction import predict_depression, show_risk, explain_prediction
from about import about_info
from model_wrapper import fine_tuned_ann_predict_wrapper
from suggestion import get_feature_advice
from shap_plot import shap_plot_waterfall
from feature_importance_plot import permutation_importance_plot

st.set_page_config("Predict Student Depression Application", layout = "wide")

page =navigation_slidebar()

if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None

if page == "Home":
    home_page()

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.divider()

    uploaded_file = st.file_uploader("Upload Student Dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        #cache everytime load homepage for reduce the memory of data preprocessing and cleaning
        @st.cache_data
        def load_cleaned(df):
            return preprocess_pipeline(df)
        
        st.session_state.df_cleaned = load_cleaned(df)
        st.success("Dataset loaded and preprocessing successfully !")

    if st.session_state.df_cleaned is not None:
        df_cleaned = st.session_state.df_cleaned
        with st.form("cleaned_dataframe_Preview"):
            st.subheader("Cleaned Student Dataset Preview")
            st.dataframe(df_cleaned)
            st.form_submit_button("Refresh Dataset Preview")

        with st.form("cleaned_dataframe_describe"):
            st.subheader("Data Overview")
            st.dataframe(df_cleaned.describe())
            st.form_submit_button("Refresh Dataset Overview")

        # Histogram Feature Selection
        with st.form("histrogram_form"):
            st.subheader("Histogram Distribution [Multi Select Features]")
            selected_features = st.multiselect(
                "Choose Features:",
                options=df_cleaned.columns.tolist(),
                default=["Age", "CGPA", "Financial Stress"]
            )

            submitted = st.form_submit_button("Show Feature Distribution Histogram Analysis")
            if submitted:
                histogram_plot(df_cleaned, selected_features)
        
        # Correlation Heatmap
        with st.form("correlation_heatmap_form"):
            st.subheader("Correlation Heatmap")
            submitted = st.form_submit_button("Show Feature Distribution Correlation Heatmap Analysis")
            if submitted:
                correlation_heatmap_plot(df_cleaned)
        
        # Boxplot Feature Selection
        with st.form("boxplot_form"):
            st.subheader("Boxplot Distribution [Multi Select Features]")
            selected_features = st.multiselect(
                "Choose Features:",
                options=df_cleaned.columns.tolist(),
                default=["Age", "CGPA", "Financial Stress"]
            )
            submitted = st.form_submit_button("Show Feature Distribution Boxplot Analysis")
            if submitted:
                boxplot(df_cleaned, selected_features)

elif page == "Factors Influencing Depression":
    st.title("Factor Depression Risk Analysis ")
    st.subheader("Barplot Analysis (Features that causes depression)")

    st.divider()

    if st.session_state.df_cleaned is None:
        st.warning("Please upload the student dataset in the EDA page first.")
    else:
        factor_risks_analysis(st.session_state.df_cleaned)  
        permutation_importance_plot()

elif page == "Models Evaluation":
    st.title("Model Performance Comparison")

    with st.form("accuracy_form"):
        st.subheader("Train vs Test Performance Accuracy")  
        plot_train_test_accuracy()
        st.form_submit_button("Refresh Model Accuracy Comparison Plot")

    with st.form("roc_auc_curve_form"):
        st.subheader("ROC AUC Curves Plot (ANN and Fine Tune ANN)")
        roc_auc_curve_plot()
        st.form_submit_button("Refresh ROC AUC Curve Plot")

elif page == "Student Depressed Prediction":       
    st.title("Student Depression Prediction System")
    st.subheader("Enter Student Personal Information")

    st.divider()

    with st.form("prediction_form"):
        gender = st.selectbox("Gender" , ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=43, value=20)
        academic_pressure = st.slider("Academic Pressure", 0, 5, value=3)
        cgpa = st.number_input("CGPA", 0.0, 4.0, step=0.01, value= 3.00)
        study_satisfaction = st.slider("Study Satisfaction", 0, 5, value= 3)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"],
            index=0
        )
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ['Healthy', 'Moderate', 'Unhealthy', 'Others'],
            index=0
        )
        study_hours = st.number_input("Study Hours", 0, 12, value=3)
        financial_stress = st.slider("Financial Stress", 1, 5, value=1)
        suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["Yes", "No"], index=1)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"], index=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        prediction, probability, X_input_scaled, raw_inputs = predict_depression(
            gender, age, academic_pressure, cgpa, study_satisfaction,
            sleep_duration, dietary_habits, suicidal_thoughts,
            study_hours, financial_stress, family_history
        )
        risk = show_risk(probability)

        if prediction == "Depressed":
         st.error(f"""
        **Prediction:** {prediction}  
        **Confidence:** {probability:.2%}  
        **Risk Level:** {risk}
        """)
        else:
         st.success(f"""
        **Prediction:** {prediction}  
        **Confidence:** {probability:.2%}  
        **Risk Level:** {risk}
        """)

        st.divider()
        st.subheader(" Factors Contributing to This Prediction")

        contributions, shap_values, feature_names, expected_value = explain_prediction(X_input_scaled)

        for feature, value in contributions[:11]:
            if value > 0:
                st.write(f"🔺 **{feature}** increased depression risk")
                feature_value = raw_inputs[feature]  # define feature_index_map
                st.write(f"💡 Advice: {get_feature_advice(feature, feature_value)}")


        st.divider()
        st.subheader("SHAP Waterfall Plot for This Prediction")
        shap_plot_waterfall(shap_values, X_input_scaled, feature_names, expected_value)

elif page == "About":
    about_info()