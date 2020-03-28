from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from bot import *

# Connect Four Game Example 
if __name__ == '__main__':
	param = Settings(RENDER=False, ACTION=0.01)

	env = ConnectFourGame(param, param["RENDER"])

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

		if episode % UPDATE_TARGET_MODEL_EVERY == 0: # and episode != 0: # If necessary update target model
			env.set_display_mode(True)
			param["RENDER"] = True
			print("Episode : ", episode)
			agent.update_target_model()
		else:
			env.set_display_mode(False)

			param["RENDER"] = False