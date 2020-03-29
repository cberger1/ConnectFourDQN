import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
from settings import Settings
from player import Player, PlayerManager
from game import ConnectFourGame


EPISODES = 1_000
UPDATE_TARGET_MODEL_EVERY = 200
REPLAY_MEMORY_SIZE = 10_000
MIN_TRAIN_SAMPLE = 100
BATCH_SIZE = 32
GAMMA = 0.98
EPSILON = 0.9
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.1

RENDER_EVERY = 50
SHOW_GAME_OVER = True
MAX_ACTIONS = 7 * 6

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

		# One Hot Encoder / Decoder
		self.codec = OneHotEncoder(self.param["ACTION_SPACE"]-1)

	def create_model(self, model=None):
		if model == None:
			model = Sequential()

			model.add(Convolution2D(8, (4,4), activation="relu", padding="same", input_shape=(7, 6, 1)))
			model.add(Flatten())
			model.add(Dense(32, activation="relu"))
			model.add(Dense(16, activation="relu"))
			model.add(Dense(self.param["ACTION_SPACE"], activation="relu"))

			model.compile(optimizer=Adam(), loss="mse")

		return model

	def play(self, state, player, use_target_model=False):
		global EPSILON
		# First compute the Q-Values, then return the index with the highest Q-Value aka the action
		if use_target_model:
			return np.argmax(self.target_model.predict(player * np.array([state]))[0])
		else:
			# Decay epsilon
			EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

			if EPSILON < random.random():
				return random.randint(0,6)
			else:
				return np.argmax(self.model.predict(player * np.array([state]))[0])

	def update_replay_memory(self, state, player, action, reward, new_state):
		self.replay_memory.append((state, player, action, reward, new_state))

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def train(self, compute_new_state=None):
		if len(self.replay_memory) < MIN_TRAIN_SAMPLE:
			return

		sample = random.sample(self.replay_memory, BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			state, player, action, reward, opponent_state = sample[i]

			opponent_player = -1 * player
			opponent_action = self.play(opponent_state, opponent_player, use_target_model=True) # Opponent makes the best possible action

			if compute_new_state != None:
				new_state = compute_new_state(opponent_state, opponent_player, opponent_action) # New State after opponent has played
			else:
				new_state = opponent_state.copy()
				if new_state[opponent_action][0] == 0: # Free spot in that column
					for row in range(5, -1, -1):
						if new_state[opponent_action][row] == 0:
							new_state[opponent_action][row] = opponent_player # Assign given value

			target = reward + GAMMA * max(self.target_model.predict(player * np.array([new_state]))) # The target Q-Value of the played action

			q_values = self.model.predict(player * np.array([state]))[0]

			for j in range(self.param["ACTION_SPACE"]):
				if j == action:
					q_values[j] == target

				if new_state[j][0] != 0: # Check if action j is not possible
					q_values[j] == self.param["UNAUTHORIZED"]

			x.append(state)
			y.append(q_values)

		self.model.fit(np.array(x), np.array(y), verbose=0)