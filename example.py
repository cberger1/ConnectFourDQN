from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerConsole, PlayerManager
from bot import AgentDQN
from keras.models import load_model
import time


MODEL_PATH = "models/8x8c-32d-16d/1585904133/v9-loss-2.0753712604796443e-13"


# Connect Four Game Example 
if __name__ == "__main__":
	DISPLAY = True

	param = Settings()
	game = ConnectFourGame(param)

	player = Player(param)
	bot = AgentDQN(param, model=load_model(MODEL_PATH))

	player_manager = PlayerManager(bot, player)
		
	state = game.get_state()

	while not game.over:
		player, action = player_manager.play(state=state) # Passing the state for the AgentDQN

		reward, new_state = game.step(player, action)
		
		state = new_state

		if DISPLAY:
			game.render()
		else:
			print(f"Player : {player}, Action : {action}")

	game.show_game_over_screen()