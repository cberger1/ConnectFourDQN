from keras.utils import Sequence
from multiprocessing import Pool, Queue, Lock
from game import ConnectFourGame


class Generator(Sequence):

	def __init__(self, brain, replay_memory, batch_size=32, batch_per_epoch=256):
		self.brain = brain
		self.replay_memory = replay_memory
		self.batch_size = batch_size
		self.batch_per_epoch = batch_per_epoch

	def __len__(self):
		# Retruns number of batches per epoch
		return self.batch_per_epoch

	def __getitem__(self, index):
		# Retruns one batch
		x = []
		y = []

		samples = random.choices(self.replay_memory, k=self.batch_size)

		with Pool() as pool:
			tasks = Queue()
			outs = Queue()

			lock = Lock()

			for sample in samples:
				tasks.put(sample)

			tasks.close()

			pool.map("prepare_sample", [tasks, outs, lock])
			
			pool.close()
			pool.join()

			for i in range(self.batch_size):
				state, q_values = outs.get()

				x.append(state)
				y.append(q_values)

			outs.close()

	def on_epoch_end(self):
		# Called at the end of each epoch
		brain.on_epoch_end()

	def prepare_sample(tasks, outs, lock):
		param = brain.param
		simulator = ConnectFourGame(param, display=False)

		while True:
			if tasks.empty():
				break
			else:
				# sample = tasks.get()

				state, player, action, reward, opponent_state, over = tasks.get()

				with lock:
					q_values = brain.model.predict(player * np.array([state]))[0]

				if over:
					target = reward
				else:
					simulator.set_state(opponent_state, over)

					opponent_player = -1 * player
					
					# Opponent makes the best possible action
					with lock:
						opponent_action = np.argmax(brain.target_model.predict(opponent_player * np.array([opponent_state]))[0])

					opponent_reward, new_state = simulator.step(opponent_player, opponent_action)

					if opponent_reward == param["WIN"]: # Opponent has won
						target = param["LOSE"]
					elif opponent_reward == param["DRAW"]: # It ended on a draw
						target = param["DRAW"]
					else:
						if opponent_reward == param["UNAUTHORIZED"] and param["END_ON_UNAUTHORIZED"]:
							# Opponent has made an unauthorized move and the game is over
							target = param["WIN"]
						else:
							# The target Q-Value of the played action
							with lock:
								target = reward + GAMMA * max(brain.target_model.predict(player * np.array([new_state]))[0])

				q_values[action] = target

				# Helping the model train faster
				for a in range(param["ACTION_SPACE"]):
					if simulator.is_action_authorizied(state, a): # Check if action a is not possible
						q_values[a] = param["UNAUTHORIZED"]

				outs.put((state, q_values))
