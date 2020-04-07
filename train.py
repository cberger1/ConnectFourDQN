from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from bot import *
import keras
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.client import device_lib
# from keras.models import load_model
import time


MODEL_DIR = f"models/{MODEL_NAME}/{int(time.time())}"

TRACK_MODEL = True

# Training Script
if __name__ == '__main__':
	print(device_lib.list_local_devices())
	# physical_devices = tf.config.experimental.list_physical_devices("GPU")
	# tf.config.experimental.set_memory_growth(physical_devices[0], True)

	display = False

	param = Settings(ACTION=-0.01)

	env = ConnectFourGame(param, display)

	agent = AgentDQN(param)
	agent_random = AgentRadnom()

	player_manager = PlayerManager(agent)

	if TRACK_MODEL:
		tensorboard = TensorBoard(log_dir=f"logs\{MODEL_NAME}-{int(time.time())}")
		tensorboard.set_model(agent.model)

	loss = 0
	epsilon = EPSILON

	for episode in range(EPISODES):
		# Checks if this episode the game is going to be rendered
		display = episode % RENDER_EVERY == 0

		env.set_display_mode(display)
		
		state = env.reset(display)
		steps = 0
		
		while not env.over and steps <= MAX_ACTIONS:
			# Decay epsilon
			epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

			player, action = player_manager.play(state=state, epsilon=epsilon)

			reward, new_state = env.step(player, action)

			agent.update_replay_memory(state, player, action, reward, new_state, env.over) # Add sample to the database of the agent
			loss += agent.train() # Will only train if enough samples are available

			state = new_state
			steps += 1

			if display:
				env.render() # Render game board
				env.pause(0.2) # Pause (in seconds)

		if display and SHOW_GAME_OVER:
			env.show_game_over_screen()

		if episode % PLOT_EVERY == 0 and episode != 0:
			if TRACK_MODEL:
				tensorboard.on_epoch_end(episode, {"loss" : loss/PLOT_EVERY})
			loss = 0

		if episode % UPDATE_TARGET_MODEL_EVERY == 0 and episode != 0: # If necessary update target model
			print("Episode : ", episode)
			agent.update_target_model()

		if episode % SAVE_EVERY == 0 and episode != 0: # If necessary save target model
			print("Saving...")
			agent.save(MODEL_DIR, f"v{episode:06}")
			print("Done")

	print("End of Training!\nSaving...")
	agent.update_target_model()
	agent.save(MODEL_DIR, f"v{EPISODES:06}")
	print("Done")

	if TRACK_MODEL:
		tensorboard.on_train_end(None)

