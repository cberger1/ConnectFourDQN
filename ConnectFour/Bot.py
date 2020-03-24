import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
from Game import *

REPLAY_MEMORY_SIZE = 10_000
BATCH_SIZE = 32
GAMMA = 0.98

UNAUTHORIZED = -1

ACTION_SPACE = 7

class OneHotEncoder:

	def __init__(self, max_arg):
		self.max_arg = max_arg # Highest argument that can be encoded

	def encode(self, arg):
		array = np.zeros((self.max_arg,)) # Array with only zeros
		array[arg] = 1 # Set element with index arg to one
		return array

	def decode(self, array):
		return np.argmax(array)

class AgentDQN(Player):

	def __init__(self, model=None, replay_memory=None):
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
		self.codec = OneHotEncoder(ACTION_SPACE-1)

	def play(self, state):
		# First compute the Q-Values, then return the index with the highest Q-Value aka the action
		print("input_shape : ", np.shape(state), "\n")
		return np.argmax(self.model.predict(state)) 

	def update_replay_memory(self, state, action, reward, new_state):
		self.replay_memory.append((state, action, reward))

	def train(self, compute_new_state):
		sample = random.sample(self.replay_memory, BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			state, action, reward, opponent_state = sample[i]

			opponent_action = np.argmax(self.target_model.predict(new_state)) # Opponent makes the best possible action

			new_state = compute_new_state(opponent_state, opponent_action) # New State after opponent has played

			target = reward + GAMMA * max(self.target_model.predict(my_new_state)) # The target Q-Value of the played action

			q_values = self.model.predict(new_state)[0]

			for j in range(ACTION_SPACE):
				if j == action:
					q_values[j] == target

				if new_state[j] != 0:
					q_values[j] == UNAUTHORIZED

			x.append(state)
			y.append(q_values)

		self.model.fit(x, y, batch=BATCH_SIZE)

	def create_model(self, model=None):
		if model == None:
			model = Sequential()

			model.add(Convolution2D(8, (4,4))) # , input_shape=(7,6)
			model.add(Flatten())
			model.add(Dense(32))
			model.add(Dense(16))
			model.add(Dense(ACTION_SPACE))

			model.compile(optimizer=Adam(), loss="mse")

		return model


	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

# Connect Four Game Example 
if __name__ == '__main__':
	env = ConnectFourGame()

	agent = AgentDQN()

	player_manager = PlayerManager(agent)

	state = env.get_state()

	while not env.over:
		current_player, action = player_manager.play(state)

		reward, new_state = env.step(ENCODE_PLAYER[current_player], current_player, action)

		agent.update_replay_memory(state, action, reward, new_state)

		state = new_state

		if RENDER:
			env.render()

	if RENDER:
		env.show_game_over_screen()