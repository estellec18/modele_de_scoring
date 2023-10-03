import joblib
import numpy as np

def predict(data):
    clf = joblib("pipeline_logreg_best.joblib")
    classes = {0:"Client ok",1:"Default"}
    preds = clf.predict_proba([data])[0]
    return (classes[np.argmax(preds)], preds)