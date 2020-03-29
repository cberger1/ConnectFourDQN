from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from bot import *


# Training Script
if __name__ == "__main__":
	display = True

	param = Settings(ACTION=0.01)

	env = ConnectFourGame(param, display)

	agent = AgentDQN(param)
	agent_random = AgentRadnom()

	player_manager = PlayerManager(agent, agent_random)

	for episode in range(EPISODES):
		# Checks if this episode the game is going to be rendered
		display = episode % RENDER_EVERY == 0

		env.set_display_mode(display)
		
		state = env.reset(display)
		num_of_actions = 0

		while not env.over and num_of_actions <= MAX_ACTIONS:
			player, action = player_manager.play(state=state)

			reward, new_state = env.step(player, action)

			agent.update_replay_memory(state, player, action, reward, new_state) # Add sample to the database of the agent
			agent.train() # Will only train if enough samples are available

			state = new_state
			num_of_actions += 1

			if display:
				env.render() # Render game board
				env.pause(0.2) # Pause (in seconds)

		if display and SHOW_GAME_OVER:
			env.show_game_over_screen()

		if episode % UPDATE_TARGET_MODEL_EVERY == 0 and episode != 0: # If necessary update target model
			print("Episode : ", episode)
			agent.update_target_model()
