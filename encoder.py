import math

import numpy as np
import tensorflow as tf
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input, Masking
from keras.models import Model, load_model

num_features = 5
# features
# 0 identity [-1, 0, 1, 2, 3]
# 1 azimuth [-pi, pi]
# 2 elevation [0, pi/2]
# 3 speed x (m/s)
# 4 speed y (m/s)
num_elements = 21
# elements
# self x1
# target x20
# obstacle x20
num_samples = 32


class Encoder():
    def __init__(self, model_filename='autoencoder/encoder.h5'):
        self.model_filename = model_filename
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def load_model(self, cpu=True):
        physical_devices = tf.config.list_physical_devices('CPU')
        print(physical_devices)
        with tf.device('/CPU:0' if cpu else '/GPU:0'):
            self.encoder = load_model(self.model_filename)
            self.decoder = load_model('autoencoder/decoder.h5')
            self.autoencoder = load_model('autoencoder/autoencoder.h5')
            self.autoencoder.compile(optimizer='adam', loss='mse')
            print(self.autoencoder.summary)
        return self

    def encode(self, inputs, cpu=True):
        with tf.device('/CPU:0' if cpu else '/GPU:0'):
            encoded = self.encoder.predict(inputs, verbose=0)[0]
        return encoded

    def create_model(self):
        inputs = Input([num_elements, num_features])
        lstm1 = LSTM(256, name='lstm1', activation='tanh', return_sequences=True)(inputs)
        lstm2 = LSTM(128, name='lstm2', activation='tanh', return_sequences=True)(lstm1)
        lstm3 = LSTM(128, name='lstm3', activation='tanh', return_sequences=False)(lstm2)
        repeat = RepeatVector(num_elements)(lstm3)
        lstm4 = LSTM(128, name='lstm4', activation='tanh', return_sequences=True)(repeat)
        lstm5 = LSTM(128, name='lstm5', activation='tanh', return_sequences=True)(lstm4)
        lstm6 = LSTM(256, name='lstm6', activation='tanh', return_sequences=True)(lstm5)
        outputs = TimeDistributed(Dense(num_features))(lstm4)

        self.autoencoder = Model(inputs=inputs, outputs=outputs)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        return self

    def train_encoder(self, num_samples=4096, epochs=10000):
        train_set = create_random_sample(num_samples)
        valid_set = create_random_sample(int(num_samples / 16))
        self.autoencoder.fit(train_set, train_set, epochs=epochs, verbose=2, validation_data=(valid_set, valid_set))
        self.encoder = Model(self.autoencoder.input, self.autoencoder.get_layer(index=3).output)
        self.decoder = Model(self.autoencoder.get_layer(index=3).output, self.autoencoder.output)
        self.autoencoder.save('autoencoder.autoencoder.h5')
        self.encoder.save('autoencoder/encoder.h5')
        self.decoder.save('autoencoder/decoder.h5')
        return self


def create_random_sample(num=5000):
    train_set = np.zeros([num, num_elements, num_features])
    self_index = 0
    target_index = 1
    obstacle_index = 11
    for step in train_set:
        num_target = np.random.randint(1, 10)
        num_obstacle = np.random.randint(1, 10)
        step[self_index] = [1, random_azimuth(), random_elevation(), random_speed(), random_speed()]
        for j in range(target_index, target_index + num_target):
            step[j] = [random_priority(), random_azimuth(), random_elevation(), random_speed(), random_speed()]
        for k in range(obstacle_index, obstacle_index + num_obstacle):
            step[k] = [-1, random_azimuth(), random_elevation(), random_speed(), random_speed()]
    return train_set


def random_priority():
    return np.random.randint(3, 5)


def random_azimuth():
    return np.random.uniform(0, math.pi * 2)


def random_elevation():
    return np.random.uniform(math.pi / 20, math.pi / 2)


def random_speed():
    return np.random.uniform(0, 1)
