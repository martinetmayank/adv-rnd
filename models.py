from tensorflow import keras
from gensim.models.word2vec import Word2Vec


WV_A = 'models\CNN_A\cnna.model'
CNN_A = 'models\CNN_A\cnn_a.h5'

WV_B = 'models\CNN_B\cnnb.model'
CNN_B = 'models\CNN_B\cnn_b.h5'


def load_models():
    MODELS = {
        'CNN_A': {
            'WV_model': Word2Vec.load(WV_A),
            'CNN_model': keras.models.load_model(CNN_A),
            'length': 1712
        },
        'CNN_B': {
            'WV_model': Word2Vec.load(WV_B),
            'CNN_model': keras.models.load_model(CNN_B),
            'length': 201
        },
    }

    return MODELS
