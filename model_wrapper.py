from keras.models import load_model

fine_tune_ann_model = load_model("saved_models/fine_tune_ann_model.h5")

def fine_tuned_ann_predict_wrapper(X):
    return fine_tune_ann_model.predict(X, verbose=0)