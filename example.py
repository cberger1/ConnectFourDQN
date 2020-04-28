import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerConsole, PlayerManager
from bot import AgentDQN
from keras.models import load_model
import time
import numpy as np


MODEL_PATH = "models/128d-128d-64d/1587831378/v024000"


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
		player, action = player_manager.play(state=state) # Passing state for the AgentDQN

		prediction = bot.model.predict(player * np.array([state]))
		print(np.round(prediction, 2))

		reward, new_state = game.step(player, action)

		# game.pause(0.5)
		
		state = np.copy(new_state)

		if DISPLAY:
			game.render()
		else:
			print(f"Player : {player}, Action : {action}")

	game.show_game_over_screen()