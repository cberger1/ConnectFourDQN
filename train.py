from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from bot import *
from keras.callbacks import TensorBoard, ModelCheckpoint
import time


# Training Script
if __name__ == '__main__':
	display = False

	param = Settings(ACTION=0.01)

	env = ConnectFourGame(param, display)

	agent = AgentDQN(param)
	agent_random = AgentRadnom()

	player_manager = PlayerManager(agent) #, agent_random)

	tensorboard = TensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
	tensorboard.set_model(agent.model)

	for episode in range(EPISODES):
		# Checks if this episode the game is going to be rendered
		display = episode % RENDER_EVERY == 0

		env.set_display_mode(display)
		
		state = env.reset(display)
		steps = 0
		
		while not env.over and steps <= MAX_ACTIONS:
			player, action = player_manager.play(state=state)

			reward, new_state = env.step(player, action)

			agent.update_replay_memory(state, player, action, reward, new_state, env.over) # Add sample to the database of the agent
			loss = agent.train() # Will only train if enough samples are available

			state = new_state
			steps += 1

			if display:
				env.render() # Render game board
				env.pause(0.2) # Pause (in seconds)

		if loss != None:
			tensorboard.on_epoch_end(episode, {"loss" : loss})

		if display and SHOW_GAME_OVER:
			env.show_game_over_screen()

		if episode % UPDATE_TARGET_MODEL_EVERY == 0 and episode != 0: # If necessary update target model
			print("Episode : ", episode)
			agent.update_target_model()

		if episode % SAVE_EVERY == 0 and episode != 0: # If necessary save target model
			print("Saving...")
			agent.save(f"models/{MODEL_NAME}/{int(time.time())}", f"v{int(episode/UPDATE_TARGET_MODEL_EVERY)}-loss-{loss}")
			print("Done")

	print("End of Training!\nSaving...")
	agent.update_target_model()
	agent.save(f"models/{MODEL_NAME}/{int(time.time())}", f"v{int(episode/UPDATE_TARGET_MODEL_EVERY)}-loss-{loss}")
	print("Done")

	tensorboard.on_train_end(None)

