import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import joblib

def plot_train_test_accuracy():
    acc = joblib.load("modelsEvaluation/model_accuracy.pkl")

    models = list(acc.keys())
    train_models = [acc[m]["train"] for m in models]
    test_models = [acc[m]["test"] for m in models]

    x = np.arange(len(models))
    width = 0.40

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_train = ax.bar(x - width/2, train_models, width, label="Train Accuracy")
    bars_test = ax.bar(x + width/2, test_models, width, label="Test Accuracy")

    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison: Deep Learning vs Machine Learning Models")
    ax.set_xticks(x)  
    ax.set_xticklabels(models)
    ax.legend()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    for bar in bars_train:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.4f}", ha="center", va="bottom")
        
    for bar in bars_test:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.4f}", ha="center", va="bottom")
        
    st.pyplot(fig)