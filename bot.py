import random
import time
import os
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from settings import Settings
from player import Player, PlayerManager
from game import ConnectFourGame


EPISODES = 2_000

UPDATE_TARGET_MODEL_EVERY = 200
SAVE_EVERY = 250
PLOT_EVERY = 10

REPLAY_MEMORY_SIZE = 10_000
MIN_TRAIN_SAMPLE = 100
BATCH_SIZE = 32

GAMMA = 0.999

EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05

RENDER_EVERY = 200
SHOW_GAME_OVER = False
MAX_ACTIONS = 7 * 6

'''
Model naming:
Conv2D : {filters}c
Dense : {units}d
Dropout : d
'''

MODEL_NAME = "16c-d-32d-16d"


class OneHotEncoder:

	def __init__(self, max_arg):
		self.max_arg = max_arg # Highest argument that can be encoded

	def encode(self, arg):
		array = np.zeros((self.max_arg,)) # Array with only zeros
		array[arg] = 1 # Set element with index arg to one
		return array

	def decode(self, array):
		return np.argmax(array)


class AgentRadnom(Player):

	def __init__(self):
		pass

	def play(self, **kwargs):
		return random.randint(0,6)


class AgentDQN(Player):
	'''
	Importent Notes

	1) When predicting an action (actually the Q-Values) from a state,
	the state is multiplied with player (either 1 or -1).
	In that way the neural network (model or target_model)
	always sees the game state in the view of player 1. So,
	the neural network does get "confused".

	About DQN

	0) My short definition : Learning by playing/exploring an environement
	using deep neural networks

	1) Q-Values : To every action for a given state we assign a Q-Value.
	The action to play is just the highest Q-Value

	2) Model vs Target Model : The "target_model" is the more "stable" model.
	In the other hand the "model" is more chaotic because usually it gets a fit
	every game step!

	3) For more infos about DQN : Google is your friend!

	'''

	def __init__(self, param, model=None, replay_memory=None):
		# Game settings
		self.param = param

		# Unstable Model
		if model != None:
			self.model = model
		else:
			self.model = self.create_model()

		# Stable Model
		self.target_model = self.create_model()
		# Syncronize taget_model with model
		self.update_target_model()

		# Replay Memory
		if replay_memory != None:
			self.replay_memory = replay_memory
		else:
			self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		
		# Simulator
		self.simulator = ConnectFourGame(param, display=False)

		# One Hot Encoder / Decoder
		self.codec = OneHotEncoder(self.param["ACTION_SPACE"]-1)

	def create_model(self, model=None):
		if model == None:
			model = Sequential()

			model.add(Convolution2D(16, (4,4), activation="relu", padding="same", input_shape=(7, 6, 1)))
			model.add(Flatten())
			model.add(Dropout(0.2))
			model.add(Dense(32, activation="relu"))
			model.add(Dropout(0.2))
			model.add(Dense(16, activation="relu"))
			model.add(Dense(self.param["ACTION_SPACE"], activation="sigmoid"))

			model.compile(optimizer=Adam(), loss="huber_loss")

		print(model.summary())

		return model

	def save(self, directory, name):
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.target_model.save(f"{directory}/{name}")

	def play(self, state, player, epsilon=0, use_target_model=False):
		global EPSILON
		# First compute the Q-Values, then return the index with the highest Q-Value aka the action
		if use_target_model:
			return np.argmax(self.target_model.predict(player * np.array([state]))[0])
		else:
			if epsilon > random.random():
				return random.randint(0,6)
			else:
				return np.argmax(self.model.predict(player * np.array([state]))[0])

	def update_replay_memory(self, state, player, action, reward, new_state, over):
		self.replay_memory.append((state, player, action, reward, new_state, over))

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def train(self):
		if len(self.replay_memory) < MIN_TRAIN_SAMPLE:
			return 0

		sample = random.sample(self.replay_memory, BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			state, player, action, reward, opponent_state, over = sample[i]

			if over:
				new_state = state
				target = reward
			else:
				opponent_player = -1 * player
				opponent_action = self.play(opponent_state, opponent_player, use_target_model=True) # Opponent makes the best possible action

				self.simulator.set_state(opponent_state)

				opponent_reward, new_state = self.simulator.step(opponent_player, opponent_action)

				if opponent_reward == self.param["WIN"]:
					reward = self.param["LOSE"]
				elif opponent_reward == self.param["DRAW"]:
					reward = self.param["DRAW"]

				target = reward + GAMMA * max(self.target_model.predict(player * np.array([new_state]))[0]) # The target Q-Value of the played action

			q_values = self.model.predict(player * np.array([state]))[0]

			q_values[action] = target

			# for j in range(self.param["ACTION_SPACE"]):
			# 	if j == action:
			# 		q_values[j] = target

			# 	if new_state[j][0] != 0: # Check if action j is not possible
			# 		q_values[j] = self.param["UNAUTHORIZED"]

			x.append(state)
			y.append(q_values)

		return self.model.train_on_batch(np.array(x), np.array(y))