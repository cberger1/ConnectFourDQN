import random
import time
import os
import numpy as np
import warnings
from collections import deque
from threading import Thread, Lock
from queue import Queue
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Reshape
# from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from settings import Settings
from player import Player, PlayerManager
from grid import Grid
from game import ConnectFourGame


EPOCHS = 1
EPISODES = 12_000

UPDATE_TARGET_MODEL_EVERY = 500
SAVE_EVERY = 1_000
PLOT_EVERY = 100

REPLAY_MEMORY_SIZE = 10_000
MIN_TRAIN_SAMPLE = 1_000 # Avoid overfitting the first houndred samples
BATCH_SIZE = 32

GAMMA = 0.95

HINT = 0.35

EPSILON = 1
EPSILON_DECAY = 0.9999
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

MODEL_NAME = "16c-d-128-128-64d"


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
		self.model = self.create_model(model)

		# Stable Model
		self.target_model = self.create_model(model)
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

			# model.add(Reshape((42,), input_shape=(7, 6, 1)))
			model.add(Convolution2D(16, (4, 4), padding="valid", input_shape=(7, 6, 1), activation="tanh"))
			# model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
			model.add(Dropout(0.2))
			# model.add(Convolution2D(16, (2, 2), padding="valid", activation="tanh"))
			# model.add(Dropout(0.2))
			model.add(Flatten())
			# model.add(Dropout(0.2))
			# model.add(Dense(64, activation="relu"))
			model.add(Dense(128, activation="relu"))
			model.add(Dense(128, activation="relu"))
			model.add(Dense(64, activation="relu"))
			model.add(Dense(self.param["ACTION_SPACE"], activation="tanh"))

			model.compile(optimizer=Adam(), loss="mse")

		print(model.summary())

		return model

	def save(self, directory, name):
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.target_model.save(os.path.join(directory, name))

	def play(self, state, player, epsilon=0, use_target_model=False):
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

	def optimize2(self):
		if len(self.replay_memory) < MIN_TRAIN_SAMPLE:
			return (0, 0, 0, 0)

		setup_start = time.time() # Start Timer

		samples = random.choices(self.replay_memory, k=BATCH_SIZE) # Get a random samples

		# [0] state, [1] player, [2] action, [3] reward, [4] opponent_state, [5] over = sample

		q_values = self.model.predict(np.array([samples[i][1] * samples[i][0] for i in range(BATCH_SIZE)]))

		predictions = self.target_model.predict(np.array([-1 * samples[i][1] * samples[i][4] for i in range(BATCH_SIZE)]))
		opponent_actions = np.argmax(predictions, axis=1)

		simulation_start = time.time()

		outs = Queue()

		threads = [Thread(target=self.simulate, args=[samples[i], opponent_actions[i], outs]) for i in range(BATCH_SIZE)]

		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

		new_states = []
		targets = []

		for _ in range(BATCH_SIZE):
			new_state, target = outs.get()
			new_states.append(new_states)
			targets.append(target)

		simulation_end = time.time()

		predictions = self.target_model.predict(np.array([samples[i][1] * samples[i][0] for i in range(BATCH_SIZE)]))
		expectations = np.amax(predictions, axis=1)

		for i in range(BATCH_SIZE):
			if targets[i] == None:
				targets[i] = samples[i][3] + GAMMA * expectations[i]

			q_values[i][samples[i][2]] = targets[i]

			# Attention: very low level; If any changes are made to the game logic please consider chaning these lines!
			for action in range(self.param["ACTION_SPACE"]):
				if samples[i][0][action][0][0] != 0:
					q_values[i][action] = self.param["UNAUTHORIZED"]

		setup_end = time.time()

		train_start = time.time()
		loss = self.model.train_on_batch(np.array([samples[i][0] for i in range(BATCH_SIZE)]), np.array(q_values))
		train_end = time.time()

		simulation_time = simulation_end - simulation_start
		setup_time = setup_end - setup_start
		train_time = train_end - train_start

		# print(f"Setup : {round(setup_time, 3)}, Training : {round(train_time, 3)}, Ratio : {round(setup_time / train_time, 3)}")

		return (loss, setup_time, train_time, simulation_time)

	def simulate2(self, sample, opponent_action, outs):
		'''
		A Grid is used for simulating the opponent
		It's chosen over a more high level ConnectFourGame approach
		Because of drastic performance improvement
		'''

		# [0] state, [1] player, [2] action, [3] reward, [4] opponent_state, [5] over = sample

		grid = Grid()

		opponent_player = -1 * sample[1] # player

		if sample[5]: # over
			new_state = sample[4] # opponent_state
			target = sample[3] # reward
		else:
			grid.set_grid(sample[4])

			if not opponent_action in grid.free_column:
				opponent_action = random.choice(grid.free_column)

			cell = grid.play_coin(opponent_player, opponent_action)

			# No need to check for UNAUTHORIZED because only valid actions can be chosen
			# if cell == None and self.param["END_ON_UNAUTHORIZED"]:
			# 	target = self.param["WIN"]

			if grid.is_winning_coin(cell, opponent_player):
				target = self.param["LOSE"]
			elif grid.is_full():
				target = self.param["DRAW"]
			else:
				target = None

			new_state = grid.get_grid()

		outs.put((new_state, target))

	def optimize(self):
		if len(self.replay_memory) < MIN_TRAIN_SAMPLE:
			return (0, 0, 0, 0)

		setup_start = time.time() # Start Timer

		samples = random.choices(self.replay_memory, k=BATCH_SIZE) # Get a random batch

		states, players, actions, rewards, opponent_states, overs = zip(*samples) # Exctract from batch

		x = np.array([players[i] * states[i] for i in range(BATCH_SIZE)])

		q_values = self.model.predict(x) # Compute the q_values

		predictions = self.target_model.predict(np.array([-1 * players[i] * opponent_states[i] for i in range(BATCH_SIZE)]))
		opponent_actions = np.argmax(predictions, axis=1)

		simulation_start = time.time()

		tasks = Queue()
		outs = Queue()

		for i in range(BATCH_SIZE):
			tasks.put((rewards[i], overs[i], opponent_states[i], -1 * players[i], opponent_actions[i]))

		threads = [Thread(target=self.simulate, args=[tasks, outs]) for _ in range(BATCH_SIZE)]

		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

		new_states = []
		targets = []
		hints = []

		for _ in range(BATCH_SIZE):
			new_state, target, hint = outs.get()
			new_states.append(new_states)
			targets.append(target)
			hints.append(hint)

		simulation_end = time.time()

		predictions = self.target_model.predict(np.array([players[i] * states[i] for i in range(BATCH_SIZE)]))
		expectations = np.amax(predictions, axis=1)

		for i in range(BATCH_SIZE):
			if targets[i] == None:
				targets[i] = rewards[i] + GAMMA * expectations[i]

			q_values[i][actions[i]] = targets[i]

			# Attention: very low level; If any changes are made to the game logic please consider chaning these lines!
			for action in range(self.param["ACTION_SPACE"]):
				if hints[i] != None:
					if hints[i] == action:
						q_values[i][action] = min(q_values[i][action] + HINT, 1)
					else:
						q_values[i][action] = max(q_values[i][action] - HINT, -1)

				if states[i][action][0][0] != 0:
					q_values[i][action] = self.param["UNAUTHORIZED"]

		setup_end = time.time()

		train_start = time.time()
		loss = self.model.train_on_batch(x, np.array(q_values))
		train_end = time.time()

		simulation_time = simulation_end - simulation_start
		setup_time = setup_end - setup_start
		train_time = train_end - train_start

		# print(f"Setup : {round(setup_time, 3)}, Training : {round(train_time, 3)}, Ratio : {round(setup_time / train_time, 3)}")

		return (loss, setup_time, train_time, simulation_time)

	def simulate(self, tasks, outs):
		'''
		A Grid is used for simulating the opponent
		It's chosen over a more high level ConnectFourGame approach
		Because of drastic performance improvement
		'''

		grid = Grid()

		reward, over, opponent_state, opponent_player, opponent_action = tasks.get()

		hint = None

		if over:
			new_state = opponent_state
			target = reward
		else:
			grid.set_grid(opponent_state)

			if not opponent_action in grid.free_column:
				opponent_action = random.choice(grid.free_column)

			cell = grid.play_coin(opponent_player, opponent_action)

			# No need to check for UNAUTHORIZED because only valid actions can be chosen
			# if cell == None and self.param["END_ON_UNAUTHORIZED"]:
			# 	target = self.param["WIN"]
			
			if grid.is_winning_coin(cell, opponent_player):
				target = self.param["LOSE"]
				hint = opponent_action
			elif grid.is_full():
				target = self.param["DRAW"]
			else:
				target = None

			new_state = grid.get_grid()

		outs.put((new_state, target, hint))

	def train(self):
		warnings.warn("This function is super slow! Please consider using optimze!")

		if len(self.replay_memory) < MIN_TRAIN_SAMPLE:
			return (0, 0, 0)

		setup_start = time.time()

		sample = random.choices(self.replay_memory, k=BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			state, player, action, reward, opponent_state, over = sample[i]

			q_values = self.model.predict(player * np.array([state]))[0]

			if over:
				target = reward
			else:
				self.simulator.set_state(opponent_state, over)

				opponent_player = -1 * player
				opponent_action = self.play(opponent_state, opponent_player, use_target_model=True) # Opponent makes the best possible action

				if not opponent_action in self.simulator.valid_actions():
					opponent_action = random.choice(self.simulator.valid_actions())

				opponent_reward, new_state = self.simulator.step(opponent_player, opponent_action)

				if opponent_reward == self.param["WIN"]: # Opponent has won
					target = self.param["LOSE"]
				elif opponent_reward == self.param["DRAW"]: # It ended on a draw
					target = self.param["DRAW"]
				else:
					if opponent_reward == self.param["UNAUTHORIZED"] and self.param["END_ON_UNAUTHORIZED"]:
						# Opponent has made an unauthorized move and the game is over
						target = self.param["WIN"]
					else:
						# The target Q-Value of the played action
						target = reward + GAMMA * max(self.target_model.predict(player * np.array([new_state]))[0])

			q_values[action] = target

			# Helping the model train faster
			for a in range(self.param["ACTION_SPACE"]):
				if self.simulator.is_action_authorizied(state, a): # Check if action a is not possible
					q_values[a] = self.param["UNAUTHORIZED"]

			x.append(player * state)
			y.append(q_values)

			setup_end = time.time()
			setup_time = setup_end - setup_start

			train_start = time.time()
			loss = self.model.train_on_batch(np.array(x), np.array(y))
			train_end = time.time()
			train_time = train_end - train_start

			# print(f"Setup : {round(setup_time, 3)}, Training : {round(train_time, 3)}, Ratio : {round(setup_time / train_time, 3)}")

		return (loss, setup_time, train_time)