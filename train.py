# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from bot import *
import time
import random
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.backend.tensorflow_backend import set_session
from keras.models import load_model


MODEL_DIR = f"models/{MODEL_NAME}/{int(time.time())}"
# MODEL_PATH = "models/16c-d-128d-128d-64d/1587478746/v024000"
TRACK_MODEL = True

# Training Script
if __name__ == '__main__':
	
	display = False

	param = Settings()

	env = ConnectFourGame(param, display)

	agent = AgentDQN(param)# , load_model(MODEL_PATH))
	# bgent = AgentDQN(param, load_model(MODEL_PATH))

	player_manager = PlayerManager(agent) # , bgent)

	if TRACK_MODEL:
		tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
		tensorboard.set_model(agent.model)

	total_loss = 0
	total_steps = 0

	# total_setup_time = 0
	# total_train_time = 0
	# total_simulation_time = 0
	for epoch in range(EPOCHS):
		epsilon = EPSILON

		for episode in range(EPISODES):
			# Checks if this episode the game is going to be rendered
			# display = episode % RENDER_EVERY == 0

			env.set_display_mode(display)

			state = env.reset(display)

			player_manager.reset()

			steps = 0
			
			# Decay epsilon
			epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

			while not env.over and steps <= MAX_ACTIONS:
				player, action = player_manager.play(state=state, epsilon=epsilon)

				if not action in env.valid_actions():
					action = random.choice(env.valid_actions())

				reward, new_state = env.step(player, action)

				agent.update_replay_memory(np.copy(state), player, action, reward, np.copy(new_state), env.over) # Add sample to the database of the agent
				loss, setup_time, train_time, simulation_time = agent.optimize()

				# if player == 1:
				# 	agent.update_replay_memory(np.copy(state), player, action, reward, np.copy(new_state), env.over) # Add sample to the database of the agent
				# 	loss, setup_time, train_time, simulation_time = agent.optimize()
				# else:
				# 	bgent.update_replay_memory(np.copy(state), player, action, reward, np.copy(new_state), env.over) # Add sample to the database of the agent
				# 	loss, setup_time, train_time, simulation_time = agent.optimize()
				
				total_loss += loss
				# total_setup_time += setup_time
				# total_train_time += train_time
				# total_simulation_time += simulation_time

				state = np.copy(new_state)
				steps += 1

				if display:
					env.render() # Render game board
					env.pause(0.2) # Pause (in seconds)

			total_steps += steps

			if display and SHOW_GAME_OVER:
				env.show_game_over_screen()

			if episode % PLOT_EVERY == 0 and episode != 0:
				if TRACK_MODEL:
					tensorboard.on_epoch_end(epoch*EPISODES + episode, {"loss" : total_loss / PLOT_EVERY, "steps" : total_steps / PLOT_EVERY})
				
				total_loss = 0
				total_steps = 0

				# setup = round(total_setup_time / PLOT_EVERY, 3)
				# train = round(total_train_time / PLOT_EVERY, 3)
				# simulation = round(total_simulation_time / PLOT_EVERY, 3)

				# if train != 0:
				# 	ratio = round(setup / train, 3)
				# else:
				# 	ratio = None

				# print(f"Setup : {setup}, Training : {train}, Ratio : {ratio}, Simulation : {simulation}")

				# total_setup_time = 0
				# total_train_time = 0
				# total_simulation_time = 0

			if episode % UPDATE_TARGET_MODEL_EVERY == 0 and episode != 0: # If necessary update target model
				print("Episode : ", epoch*EPISODES + episode)
				agent.update_target_model()
				# bgent.update_target_model()

			if episode % SAVE_EVERY == 0 and episode != 0: # If necessary save target model
				print("Saving...")
				agent.save(MODEL_DIR, f"v{epoch*EPISODES + episode:06}")
				print("Done")

	print("End of Training!\nSaving...")
	agent.update_target_model()
	agent.save(MODEL_DIR, f"v{EPOCHS*EPISODES:06}")
	print("Done")

	if TRACK_MODEL:
		tensorboard.on_train_end(None)