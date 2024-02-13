import pygame
import random

# Initialize pygame
pygame.init()

# Define colors
def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

# Set up display
width, height = 640, 480
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Colorful Snake Game')

# Set up game clock
clock = pygame.time.Clock()

# Set up game parameters
snake_size = 10
snake_speed = 15

# Snake function
def draw_snake(snake_size, snake_body):
    for pixel in snake_body:
        pygame.draw.rect(game_display, pixel[2], [pixel[0], pixel[1], snake_size, snake_size])

# Main function
def run_game():
    game_over = False
    game_close = False

    x1 = width / 2
    y1 = height / 2

    x1_change = 0
    y1_change = 0

    snake_body = []
    snake_length = 1

    foodx = round(random.randrange(0, width - snake_size) / 10.0) * 10.0
    foody = round(random.randrange(0, height - snake_size) / 10.0) * 10.0

    while not game_over:

        while game_close:
            game_display.fill((255, 255, 255))
            font_style = pygame.font.SysFont(None, 50)
            message = font_style.render('You Lost! Press Enter to Play Again', True, (0, 0, 0))
            game_display.blit(message, [width / 6, height / 3])
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        run_game()

                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_size
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_size
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_size
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_size
                    x1_change = 0

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        game_display.fill((0, 0, 0))
        pygame.draw.rect(game_display, (255, 0, 0), [foodx, foody, snake_size, snake_size])
        snake_head = [x1, y1, get_random_color()]
        snake_body.append(snake_head)

        if len(snake_body) > snake_length:
            del snake_body[0]

        for block in snake_body[:-1]:
            if block[0] == snake_head[0] and block[1] == snake_head[1]:
                game_close = True

        draw_snake(snake_size, snake_body)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - snake_size) / 10.0) * 10.0
            foody = round(random.randrange(0, height - snake_size) / 10.0) * 10.0
            snake_length += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

run_game()
