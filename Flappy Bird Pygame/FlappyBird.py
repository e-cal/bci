import pygame
from pygame.locals import *
import random
import time


pygame.init()
clock = pygame.time.Clock()

screen = pygame.display.set_mode((650, 1000))

background = pygame.image.load('background.png')
background = pygame.transform.scale_by(background, 1.3)

screen.fill((255,255,255))

screen.blit(background, (0,0))

pygame.display.flip()

pygame.display.set_caption('Flappy Bird')

font = pygame.font.Font('freesansbold.ttf', 32)
startLine1 = font.render('Press Any Key To Start', True, (0,0,0), (114,200,207))
startLine2 = font.render('Jump (Space)', True, (0,0,0), (114,200,207))

endLine = font.render('Press Any Key To Play Again', True, (0,0,0), (114,200,207))

start = False

# Starting the game
def start_menu(start):
    while not start:
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                start = True
        
        screen.blit(startLine1, (140,400))
        screen.blit(startLine2, (220, 440))

        pygame.display.update()

    running()


def running():
    bird = pygame.image.load('bird1.png')
    pos_x = 100
    pos_y = 350
    gravity = 3
    jump = -120
    bird_rect = bird.get_rect(center=(pos_x, pos_y))

    pipe_surface = pygame.image.load('pipe.png')
    pipe_list = []
    SPAWNPIPE = pygame.USEREVENT
    pygame.time.set_timer(SPAWNPIPE, 1200)
    pipe_x = 950
    pipe_height = [400, 600, 800]

    background_pos_x = 0

    end = False

    while not end:
        for event in pygame.event.get():
            
            if event.type == QUIT:
                pygame.quit()

            # Jumping Functionality (Space Bar)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_rect = bird.get_rect(center=(pos_x, pos_y))
                if pos_y >= 0 and pos_y <= 60:
                    pos_y = 0

                else:
                    pos_y += jump
            
            if event.type == SPAWNPIPE:
                random_pipe_pos = random.choice(pipe_height)
                top_pipe = pipe_surface.get_rect(midbottom = (pipe_x, random_pipe_pos-300))
                bottom_pipe = pipe_surface.get_rect(midtop = (pipe_x, random_pipe_pos)) 
                pipe_list.extend((top_pipe, bottom_pipe))

        # Keeping only one copy of bird on screen
        screen.fill((255,255,255))


        # Checking Boundaries
        if pos_y >= 885:
            end = True

        else:
            pos_y += gravity

        background_pos_x -= 1

        screen.blit(background, (background_pos_x, 0))
        screen.blit(background, (background_pos_x + 650, 0))

        if background_pos_x < -650:
            background_pos_x = 0

        for pipe in pipe_list:
            if pipe.centerx <= 0:
                pipe_list.remove(pipe)
            pipe.centerx -= 2

        for pipe in pipe_list:
            if pipe.bottom >= 900:
                screen.blit(pipe_surface, pipe)

            else:
                flip_pipe = pygame.transform.flip(pipe_surface, False, True)
                screen.blit(flip_pipe, pipe)

        for pipe in pipe_list:
            if bird_rect.colliderect(pipe):
                end = True

        screen.blit(bird, (pos_x, pos_y))

        pygame.display.update()

        clock.tick(150)

    end_screen()


def end_screen():
    while True:
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                running()
            
        screen.blit(background, (0,0))

        screen.blit(endLine, (105, 450))

        pygame.display.update()


start_menu(start)
