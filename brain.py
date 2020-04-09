import os
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam


EPISODES = 3_000

UPDATE_TARGET_MODEL_EVERY = 200
SAVE_EVERY = 200
PLOT_EVERY = 10

REPLAY_MEMORY_SIZE = 10_000
MIN_TRAIN_SAMPLE = 1_000 # Avoid overfitting the first houndred samples
BATCH_SIZE = 32

GAMMA = 0.95

EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.1

RENDER_EVERY = 200
SHOW_GAME_OVER = False
MAX_ACTIONS = 7 * 6

'''
Model naming:
Conv2D : {filters}c
MaxPooling2D : m
Dense : {units}d
Dropout : d
'''

MODEL_NAME = "8c-d-32d-d-16d"


class Brain:

	def __init__(self, param, model=None):
		# Game settings
		self.param = param

		# Unstable Model
		self.model = self.create_model(model)

		# Stable Model
		self.target_model = self.create_model(model)
		# Syncronize taget_model with model
		self.update_target_model()

	def create_model(self, model=None):
		if model == None:
			model = Sequential()

			model.add(Convolution2D(8, (4, 4), padding="valid", input_shape=(7, 6, 1), activation="relu"))
			# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
			model.add(Flatten())
			model.add(Dropout(0.2))
			model.add(Dense(32, activation="relu"))
			model.add(Dropout(0.2))
			# model.add(Dense(32, activation="relu"))
			model.add(Dense(16, activation="relu"))
			model.add(Dense(self.param["ACTION_SPACE"], activation="tanh"))

			model.compile(optimizer=Adam(), loss="mse")

		print(model.summary())

		return model

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def save(self, directory, name):
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.target_model.save(f"{directory}/{name}")

	def train(self, batch_size, batch_per_epoch, epochs):
		self.model.fit_generator()

	def on_epoch_end(self):
		self.update_target_model()

