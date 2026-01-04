import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

ann_model = joblib.load("saved_models/ann_model.pkl")
fine_tune_ann_model = load_model("saved_models/fine_tune_ann_model.h5")

X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

def roc_auc_curve_plot():
    y_pred_probs_ann = ann_model.predict(X_test).ravel()  
    y_pred_probs_fine = fine_tune_ann_model.predict(X_test).ravel()  
    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_probs_ann)
    auc_ann = roc_auc_score(y_test, y_pred_probs_ann)

    fpr_fine, tpr_fine, _ = roc_curve(y_test, y_pred_probs_fine)
    auc_fine = roc_auc_score(y_test, y_pred_probs_fine)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {auc_ann:.4f})', color='blue')
    plt.plot(fpr_fine, tpr_fine, label=f'Fine-Tune ANN (AUC = {auc_fine:.4f})', color='green')
    plt.plot([0,1], [0,1], linestyle='--', color='red', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')

    st.pyplot(plt)