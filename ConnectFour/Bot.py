import numpy as np
import random
from collections import deque

REPLAY_MEMORY_SIZE = 10_000
BATCH_SIZE = 32
GAMMA = 0.98

ACTION_SPACE = 7

class OneHotEncoder:

	def __init__(self, max_arg)
		self.max_arg = max_arg # Highest argument that can be encoded

	def encode(self, arg):
		array = np.zeros((self.max_arg,)) # Array with only zeros
		array[arg] = 1 # Set element with index arg to one
		return array

	def decode(self, array):
		return np.argmax(array)

class AgentDQN:

	def __init__(self, target_model=None, replay_memory=None):
		# Stable Model
		if target_model != None:
			self.target_model = target_model
		else:
			self.target_model = self.create_model()

		# Unstable Model
		self.model = self.copy_model(self.target_model)

		# Replay Memory
		if replay_memory != None:
			self.replay_memory = replay_memory
		else:
			self.replay_memory = deque()

		self.replay_memory.maxlen = REPLAY_MEMORY_SIZE

		# One Hot Encoder / Decoder
		self.codec = OneHotEncoder(ACTION_SPACE-1)

	def play(self, state)
		q_values = self.model.predict(state)
		return np.argmax(q_values)

	def target_play(self, state):
		q_values = self.traget_model.predict(state)
		return np.argmax(q_values)

	def update_replay_memory(self, state, action, reward):
		self.replay_memory.append((state, action, reward))

	def train(self, compute_new_state):
		sample = random.sample(self.replay_memory, BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			my_state, my_action, reward = *sample[i]

			opponent_state = compute_new_state(my_state, my_action)
			opponent_action = self.target_play(opponent_state)

			my_new_state = compute_new_state(opponent_state, opponent_action)

			target = reward + GAMMA * max(self.target_model.predict(my_new_state))

			x.append(my_state)
			y.append(target)

		self.model.fit(x, y, batch=BATCH_SIZE)

	def create_model(self, model):
		pass

	def copy_model(self, model):
		pass

