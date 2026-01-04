import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def histogram_plot(df, selected_features):
    if len(selected_features) == 0:
        st.warning("Please select at least one feature to plot")
        return
    cols_per_row = 3
    rows = (len(selected_features) // cols_per_row) + 1
    fig, axes = plt.subplots(rows, cols_per_row, figsize =(25, rows* 4))
    axes = axes.flatten()

    for i, col in enumerate(selected_features):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    st.pyplot(fig)
    plt.close(fig)

def correlation_heatmap_plot(df):
    fig, ax =plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    plt.close(fig)  

def boxplot(df, selected_features):
    if len(selected_features) == 0:
        st.warning("Please select at least one feature to plot")
        return
    cols_per_row = 3
    rows = (len(selected_features) // cols_per_row) + 1
    fig, axes = plt.subplots(rows, cols_per_row, figsize =(25, rows* 4))
    axes = axes.flatten()

    for i, col in enumerate(selected_features):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    st.pyplot(fig)
    plt.close(fig)