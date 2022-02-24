import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
filename = 'finalized_model.sav'


# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


# DataFlair - Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
# DataFlair - Emotions to observe
observed_emotions = ['calm', 'happy',  'fearful', 'disgust']


# DataFlair - Load the data and extract features for each sound file
# def load_data(test_size=0.2):
#     x, y = [], []
#     files = glob.glob("DataFlair/ravdess data/Actor_*/*.wav")
#     for i, file in enumerate(files):
#         file_name = os.path.basename(file)
#         emotion = emotions[file_name.split("-")[2]]
#         if emotion not in observed_emotions:
#             continue
#         feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
#         print("Progress: {:.2f}%".format((i/len(files)) * 100))
#         x.append(feature)
#         y.append(emotion)
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# def train():
#     x_train, x_test, y_train, y_test = load_data(test_size=0.1)
#     print((x_train.shape[0], x_test.shape[0]))
#     print(f'Features extracted: {x_train.shape[1]}')
#     model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
#                           max_iter=1000)
#     # DataFlair - Train the model
#     model.fit(x_train, y_train)
#     pickle.dump(model, open(filename, 'wb'))
#     loaded_model = pickle.load(open(filename, 'rb'))
#     y_pred = loaded_model.predict(x_test)
#     print("Model Trained")
#     acc = accuracy_score(y_true=y_test, y_pred=y_pred)
#     print("Accuracy: {:.2f}%".format(acc * 100))


# def recorded_test(file):
#     loaded_model = pickle.load(open(filename, 'rb'))
#     feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
#     feature = list(feature)
#     x = np.array(feature)
#     feature_predict = loaded_model.predict(x)
#     print(feature_predict)
