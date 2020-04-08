from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerConsole, PlayerManager
from bot import AgentDQN
from keras.models import load_model
import time
import numpy as np


MODEL_PATH = "models/16c-d-32d-d-32d-16d/1586282246/v002000"


# Connect Four Game Example 
if __name__ == "__main__":
	DISPLAY = True

	param = Settings()
	game = ConnectFourGame(param)

	player = Player(param)
	bot = AgentDQN(param, model=load_model(MODEL_PATH))

	player_manager = PlayerManager(player, bot)
		
	state = game.get_state()

	while not game.over:
		prediction = bot.model.predict(np.array([state]))
		print(np.round(prediction, 2))

		player, action = player_manager.play(state=state) # Passing the state for the AgentDQN

		reward, new_state = game.step(player, action)
		
		state = new_state

		if DISPLAY:
			game.render()
		else:
			print(f"Player : {player}, Action : {action}")

	game.show_game_over_screen()