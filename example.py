from settings import Settings
from game import ConnectFourGame
from player import Player, PlayerManager
from grid import Grid
import pygame
import time

grid = Grid()

print(int(grid[(6,5)]))

# time.sleep(2)
# pygame.init()
# pygame.display.set_caption("ConnectFourGame")
# pygame.display.set_icon(pygame.image.load('Sprites/icon.png'))
# pygame.display.set_mode((800,800))
# clock = pygame.time.Clock()
# over = False

# while not over:
# 	clock.tick(30) # Show at most 30 FPS

# 	# Handle events
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			over = True

# pygame.quit()
# print("sleeping...")
# time.sleep(3)

# #pygame.init()
# pygame.display.set_mode((800,800))
# over = False
# while not over:
# 	clock.tick(30) # Show at most 30 FPS

# 	# Handle events
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			over = True

# pygame.quit()
# print("sleeping...")
# time.sleep(3)





# # Connect Four Game Example 
# if __name__ == '__main__':
# 	param = Settings(RENDER=True)

# 	game = ConnectFourGame(param)

# 	player_manager = PlayerManager(Player(param))

# 	while not game.over:
# 		current_player, action = player_manager.play()

# 		value = param["ENCODE_PLAYER"][current_player]

# 		reward, new_state = game.step(value, current_player, action)
		
# 		if param["RENDER"]:
# 			game.render()

# 	game.show_game_over_screen()