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
UPDATE_TARGET_MODEL_EVERY = 10
REPLAY_MEMORY_SIZE = 10_000
BATCH_SIZE = 32
GAMMA = 0.98
MAX_ACTIONS = 7 * 6

ACTION_SPACE = 7

UNAUTHORIZED = -1

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

	def play(self, *args, **kwargs):
		return random.randint(0,6)


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
		return np.argmax(self.model.predict(np.reshape(state, (1,7,6,1))))

	def update_replay_memory(self, state, action, reward, new_state):
		self.replay_memory.append((state, action, reward, new_state))

	def train(self, value, compute_new_state=None):
		if len(self.replay_memory) < BATCH_SIZE:
			return

		sample = random.sample(self.replay_memory, BATCH_SIZE)

		x = []
		y = [] 

		for i in range(BATCH_SIZE):
			state, action, reward, opponent_state = sample[i]

			opponent_action = np.argmax(self.target_model.predict(np.reshape(opponent_state, (1, 7, 6, 1)))[0]) # Opponent makes the best possible action

			if compute_new_state != None:
				new_state = compute_new_state(opponent_state, opponent_action, value) # New State after opponent has played
			else:
				new_state = opponent_state.copy()
				if new_state[opponent_action][0] == 0: # Free spot in that column
					for row in range(5, -1, -1):
						if new_state[opponent_action][row] == 0:
							new_state[opponent_action][row] = value # Assign given value

			target = reward + GAMMA * max(self.target_model.predict(np.reshape(new_state, (1, 7, 6, 1)))) # The target Q-Value of the played action

			q_values = self.model.predict(np.reshape(new_state, (1, 7, 6, 1)))[0]

			for j in range(ACTION_SPACE):
				if j == action:
					q_values[j] == target

				if new_state[j][0] != 0: # Check if action j is not possible
					q_values[j] == UNAUTHORIZED

			x.append(state)
			y.append(q_values)

		self.model.fit(np.array(x), np.array(y), verbose=0)

	def create_model(self, model=None):
		if model == None:
			model = Sequential()

			model.add(Convolution2D(8, (4,4), padding="same", input_shape=(7, 6, 1))) # , input_shape=(7,6)
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
	param = Settings(RENDER=True, ACTION=0.01)

	env = ConnectFourGame(param)

	agent = AgentDQN()
	randBot = AgentRadnom()

	player_manager = PlayerManager(agent, randBot)

	for episode in range(EPISODES):
		state = env.reset()

		num_of_actions = 0

		while not env.over and num_of_actions <= MAX_ACTIONS:
			current_player, action = player_manager.play(state)
			value = param["ENCODE_PLAYER"][current_player]

			reward, new_state = env.step(value, current_player, action)

			if current_player == 1: # Transpose --> Bot is always Player 1
				transpose = -1
			else:
				transpose = 1

			agent.update_replay_memory(transpose*state, action, reward, new_state) # Add sample to database of the agent
			agent.train(value) # Will only train if enough samples are available

			state = new_state
			num_of_actions += 1

			if param["RENDER"]:
				env.render() # Render game board
				env.pause(0.2) # Pause (in seconds)

		if param["RENDER"]:
			env.show_game_over_screen()

		if episode % UPDATE_TARGET_MODEL_EVERY == 0 and episode != 0: # If necessary update target model
			param["RENDER"] = True
			print("Episode : ", episode)
			agent.update_target_model()
		else:
			param["RENDER"] = False
