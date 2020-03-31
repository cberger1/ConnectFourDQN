from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager

# Connect Four Game Example 
if __name__ == "__main__":
	DISPLAY = True

	param = Settings()
	game = ConnectFourGame(param, DISPLAY)
	player_manager = PlayerManager(Player(param))

	while not game.over:
		player, action = player_manager.play()

		reward, new_state = game.step(player, action)
		
		if DISPLAY:
			game.render()
		else:
			print(f"Player : {player}, Action : {action}")

	game.show_game_over_screen()