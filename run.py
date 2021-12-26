import tensorflow.keras.backend as K
from nltk.tokenize import RegexpTokenizer
import numpy as np


from models import load_models
MODELS = load_models()


def evaluate_phrase(tweet, model, word2vec, max_tweet_length, vector_size=300):
    tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
    tokens = [t for t in tkr.tokenize(tweet) if not t.startswith('@')]

    vectorized_tweet = np.zeros(
        (1, max_tweet_length, vector_size), dtype=K.floatx())

    for i, token in enumerate(tokens):
        if token in word2vec.wv.key_to_index:
            vectorized_tweet[0][i] = word2vec.wv.__getitem__(token)

    prediction = model.predict(vectorized_tweet)
    print("NS: {:.2f}%".format(
        prediction[0][0]*100) + " S: {:.2f}%".format(prediction[0][1]*100))
    print("Non sarcasm" if prediction[0][0] > prediction[0][1] else "Sarcasm")

    print(prediction[0])

    s = (prediction[0][0]*100)
    ns = (prediction[0][1]*100)
    r = 'Non Sarcastic' if ns > s else 'Sarcastic'

    result = [
        r,
        (str(s)[:5], str(ns)[:5])
    ]
    return result


def predict(sentence):
    result = list()
    for model in MODELS:
        print(model)
        output = evaluate_phrase(
            sentence,
            model=MODELS[model]['CNN_model'],
            word2vec=MODELS[model]['WV_model'],
            max_tweet_length=MODELS[model]['length']
        )
        result.append([model, output])

    return result
