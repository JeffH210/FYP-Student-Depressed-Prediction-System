import streamlit as st

def home_page():
    st.title("Student Depression Prediction System")
    st.caption("A Data-Driven Mental Health Analysis Tool")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Overview 
        This application helps analyzing student mental status and predicts depression risk using machine learning and deep learning models.
        
        ### Key Features
        - Exploratory Data Analysis
        - Factors Influencing Depression
        - Depression Prediction (Fine Tune ANN) 
        - Model Comparison (DT, RF, SVM, ANN， Fine-Tuned ANN)          
        """)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        st.metric("ML Model Used to perform Train-Test Split", "4")
        st.metric(
            "Dataset Status",
            "Loaded Successfully" if st.session_state.df_cleaned is not None else "Not Loaded"
        )    
    
    st.divider()

    with st.expander(" Why This System Matters "):
        st.write(""" 
        Student depression is an increasing concern in academic environments.
        Early Detection allows institutions to provide timely support and 
        improve student's mental health and academic performance.
        """)

    st.info("Use Navigation Slidebar to select different Pages through the application!")