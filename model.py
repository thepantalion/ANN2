import os
import tensorflow as tf

class model:
    def __init__(self, path):
        # Load the models
        self.model = tf.keras.models.load_model(os.path.join(path, 'model.keras'))

    def predict(self, X, categories):
        out = self.model.predict(X)
        return out

