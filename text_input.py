import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

filename_imdb = 'finalized_model_imdb.sav'

def custom_text_check(input_text):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open("feature.pkl", "rb")))
    loaded_model = pickle.load(open(filename_imdb, 'rb'))

    test = []
    test.append(input_text)
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(test))
    predLabel = loaded_model.predict(tfidf)
    tags = ["Negative", "Positive"]
    return tags[predLabel[0]]


# def main():
#     # Custom Test: Test a review on the best performing model (Logistic Regression)
#     transformer = TfidfTransformer()
#     loaded_vec = CountVectorizer(decode_error="replace",
#                                  vocabulary=pickle.load(open("/models/feature.pkl", "rb")))
#     loaded_model = pickle.load(open(filename_imdb, 'rb'))

#     print('\nTest a custom review message')
#     c = 'y'
#     while c == 'y':
#         print('Enter review to be analysed: ', end=" ")
#         test = []
#         test.append(input())
#         tfidf = transformer.fit_transform(loaded_vec.fit_transform(test))
#         predLabel = loaded_model.predict(tfidf)
#         tags = ['Negative', 'Positive']

#         print('The review is predicted', tags[predLabel[0]])
#         print('Want to test more reviews?(y/n)')
#         c = input()
#         if c == 'n':
#             break
