import tensorflow as tf
from keras.models import load_model

model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

model = load_model(model_path)

model.summary()