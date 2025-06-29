from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

class Model:

    def __init__(self, input_shape, action_space):

        self.X_input = Input(input_shape)
        self.X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(self.X_input)
        self.X = Dense(256, activation="relu", kernel_initializer='he_uniform')(self.X)
        self.X = Dense(64, activation="relu", kernel_initializer='he_uniform')(self.X)
        self.X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(self.X)

        self.model = Model(inputs = self.X_input, outputs = self.X, name='DQN')
        self.model.compile(loss="mse", optimizer=tf.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        self.model.summary()