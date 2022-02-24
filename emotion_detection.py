import pickle
import glob
import os
import numpy as np
import pandas as pd
from train import extract_feature

filename = 'finalized_model.sav'


def custom_audio_check():
    loaded_model = pickle.load(open(filename, 'rb'))
    file = "*.wav"
    files = glob.glob(file)
    x, y = [], []
    for file in files:
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    x = np.array(x)
    feature_predict = loaded_model.predict(x)
    print(feature_predict[0])
    return feature_predict[0]


# def main():
#     loaded_model = pickle.load(open(filename, 'rb'))
#     file = "DataFlair/*.wav"
#     files = glob.glob(file)
#     x, y = [], []
#     for file in files:
#         feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#     x = np.array(x)
#     feature_predict = loaded_model.predict(x)
#     results = pd.DataFrame({"Filename": files, "Predictions": feature_predict})
#     print(results)
