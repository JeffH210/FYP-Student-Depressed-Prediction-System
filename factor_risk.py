import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def factor_risks_analysis(df_cleaned):
    st.write(
        "This section will explore how different academic and personal lifestyle "
        "influence student depression probability"
    )

    factor = st.selectbox(
        "Select a factor to analyze:",
        [
            "Age",
            "Academic Pressure",
            "CGPA",
            "Study Satisfaction",
            "Sleep Duration",
            "Dietary Habits",
            "Study Hours",
            "Financial Stress",
            "Suicidal Thoughts",
            "Family History of Mental Illness"
        ]
    )

    depression_rate = (
        df_cleaned
        .groupby(factor)["Depression"]
        .mean()
        .reset_index()
    )
    depression_rate["Depression Rate (%)"] = depression_rate["Depression"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data = depression_rate,
        x = factor,
        y = "Depression Rate (%)",
        ax = ax
    )
    ax.set_title(f"Depression Rate by {factor}")
    ax.set_ylabel("Depression Rate (%)")
    ax.set_xlabel(factor)

    st.pyplot(fig)
    plt.close(fig)

    highest = depression_rate.loc[
        depression_rate["Depression Rate (%)"].idxmax()
    ]

    lowest = depression_rate.loc[
        depression_rate["Depression Rate (%)"].idxmin()
    ]

    st.markdown("Factor Risk Interpretation")
    st.write(
        f"Student with **{highest[factor]}** exhibit the **highest depression rate** "
        f"at **{highest['Depression Rate (%)']:.1f}%**, whereas student with "
        f"**{lowest[factor]}** show the **lowest depression rate** "
        f"at **{lowest['Depression Rate (%)']:.1f}%**."
    )

    st.info(
        "This suggests that improvements in this factor may help reduce the risk "
        "of depression among students."
    )