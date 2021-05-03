import pickle
import glob
import os
import numpy as np
import pandas as pd
from train import extract_feature, train
filename = 'G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/models/finalized_model.sav'


def custom_audio_check(audio_file):
    loaded_model = pickle.load(open(filename, 'rb'))
    feature = extract_feature(
        str(audio_file), mfcc=True, chroma=True, mel=True)
    feature = list(feature)
    x = np.array(feature)
    feature_predict = loaded_model.predict(x)
    return feature_predict


def main():
    loaded_model = pickle.load(open(filename, 'rb'))
    file = "DataFlair/*.wav"
    files = glob.glob(file)
    x, y = [], []
    for file in files:
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    x = np.array(x)
    feature_predict = loaded_model.predict(x)
    results = pd.DataFrame({"Filename": files, "Predictions": feature_predict})
    print(results)
