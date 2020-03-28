from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from grid import Grid
import pygame
import time


# Connect Four Game Example 
if __name__ == '__main__':
	param = Settings(RENDER=True)

	game = ConnectFourGame(param, param["RENDER"])

	player_manager = PlayerManager(Player(param))

	while not game.over:
		current_player, action = player_manager.play()

		value = param["ENCODE_PLAYER"][current_player]

		reward, new_state = game.step(value, current_player, action)
		
		if param["RENDER"]:
			game.render()

	game.show_game_over_screen()