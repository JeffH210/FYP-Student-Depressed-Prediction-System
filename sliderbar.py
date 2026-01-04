import streamlit as st

def navigation_slidebar():
    with st.sidebar:

        st.markdown("---")
        st.markdown("### Navigation")
        
        page = st.selectbox(
            "Go To",
            [
                "Home", 
                "EDA", 
                "Factors Influencing Depression", 
                "Models Evaluation", 
                "Student Depressed Prediction", 
                "About"
            ],
            index=0
        )

        st.markdown("---")
        st.caption("Student Mental Health Dashboard v1.0")
        
    return page