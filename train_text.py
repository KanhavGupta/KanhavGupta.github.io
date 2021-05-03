import pickle
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

filename_imdb = 'G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/models/finalized_model_imdb.sav'
filename_opinions = 'G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWeb App/models/finalized_model.sav'

path = "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/data/IMDB Dataset.csv"
# path = 'data/opinions.tsv'
if path == "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/data/IMDB Dataset.csv":
    data = pd.read_csv(path, header=None, skiprows=1,
                       names=['Review', 'Sentiment'])
elif path == "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/data/opinions.csv":
    data = pd.read_table(path, header=None, skiprows=1,
                         names=['Sentiment', 'Review'])
X = data.Review
y = data.Sentiment


def train():
    # Using CountVectorizer to convert text into tokens/features
    vect = CountVectorizer(stop_words='english',
                           ngram_range=(1, 1), max_df=.80, min_df=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.1)
    # Using training data to transform text into counts of features for each message
    vect.fit(X_train)
    X_train_transform = vect.transform(X_train)
    X_test_transform = vect.transform(X_test)

    # Accuracy using Naive Bayes Model
    NB = MultinomialNB()
    NB.fit(X_train_transform, y_train)
    y_pred = NB.predict(X_test_transform)
    print('\nNaive Bayes')
    print("Accuracy Score:", metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using Logistic Regression Model
    LR = LogisticRegression()
    LR.fit(X_train_transform, y_train)
    if path == "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/data/IMDB Dataset.csv":
        pickle.dump(LR, open(filename_imdb, 'wb'))
        pickle.dump(vect.vocabulary_, open(
            "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/models/feature.pkl", "wb"))
    elif path == "G:/NIT/1.PROJECT WORK NIT/SentimentAnalysisWebApp/data/opinions.csv":
        pickle.dump(LR, open(filename_opinions, 'wb'))
    else:
        print("model not trained")
    y_pred = LR.predict(X_test_transform)
    print('\nLogistic Regression')
    print("Accuracy Score: ", metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using SVM Model
    SVM = LinearSVC()
    SVM.fit(X_train_transform, y_train)
    y_pred = SVM.predict(X_test_transform)
    print('\nSupport Vector Machine')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

    # Accuracy using KNN Model
    KNN = KNeighborsClassifier(n_neighbors=200)
    KNN.fit(X_train_transform, y_train)
    y_pred = KNN.predict(X_test_transform)
    print('\nK Nearest Neighbors (NN = 200)')
    print('Accuracy Score: ', metrics.accuracy_score(
        y_test, y_pred)*100, '%', sep='')
    print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')
