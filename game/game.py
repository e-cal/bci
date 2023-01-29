import pygame
from pygame.locals import *
import random
import time

JUMP = 20
GRAVITY = 0.1

MAX_VEL = 5

# Init game
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((650, 1000))
font = pygame.font.Font("freesansbold.ttf", 32)
background = pygame.image.load("game/background.png")
background = pygame.transform.scale(
    background, [background.get_width() * 1.3, background.get_height() * 1.3]
)
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)


class Bird:
    def __init__(
        self,
        x=100,
        y=screen.get_height() // 2,
        size=[80, 80],
        vel=0,
        img="game/brain.png",
    ) -> None:
        self.img = pygame.image.load(img)
        self.img = pygame.transform.scale(self.img, size)
        self.x = x
        self.y = y
        self.size = size
        self.vel = vel

    def get_rect(self):
        return self.img.get_rect(
            center=(self.x + (self.size[0] // 2), self.y + (self.size[1] // 2))
        ).inflate(-40, -40)

    def jump(self):
        self.vel -= JUMP

    def update(self):
        self.vel += GRAVITY

        if abs(self.vel) > MAX_VEL:
            self.vel = MAX_VEL * (abs(self.vel) / self.vel)

        self.y += self.vel

    def draw(self):
        screen.blit(self.img, (self.x, self.y))


class Pipe:
    def __init__(self) -> None:
        self.img = pygame.image.load("game/pipe.png")
        self.imginv = pygame.transform.flip(self.img, False, True)

        self.x = screen.get_width() + 50

        self.mid = random.randint(400, 800)
        self.gap = random.randint(250, 300)
        self.update()

    def collide(self, bird):
        return bird.colliderect(self.top) or bird.colliderect(self.bottom)

    def update(self):
        self.x -= 2
        self.top = self.img.get_rect(
            midbottom=(self.x, self.mid - (self.gap // 2) - 120)
        )
        self.bottom = self.img.get_rect(
            midtop=(self.x, self.mid + (self.gap // 2) - 120)
        )

    def draw(self):
        screen.blit(self.img, self.bottom)
        screen.blit(self.imginv, self.top)


def menu():
    text = font.render("Press Any Key To Start", True, (0, 0, 0), (114, 200, 207))

    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                else:
                    run()

        screen.blit(background, (0, 0))
        screen.blit(text, (105, 450))
        pygame.display.update()


def update_background(bg_x):
    bg_x -= 1
    screen.blit(background, (bg_x, 0))
    screen.blit(background, (bg_x + 650, 0))

    if bg_x < -650:
        bg_x = 0

    return bg_x


def run():
    bird = Bird()
    pipes = []

    # pipe_height = [400, 600, 800]

    bg_x = 0

    end = False

    while not end:
        screen.fill((255, 255, 255))

        clock.tick(150)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    bird.jump()

                if event.key == pygame.K_q:
                    pygame.quit()

            if event.type == SPAWNPIPE:
                pipes.append(Pipe())

        # Checking Boundaries
        if bird.y >= screen.get_height() or bird.y <= -40:
            end = True

        bg_x = update_background(bg_x)

        for pipe in pipes:
            if pipe.x <= 0:
                pipes.remove(pipe)

            pipe.update()
            pipe.draw()

            if pipe.collide(bird.get_rect()):
                end = True

        bird.update()
        bird.draw()
        pygame.display.update()


if __name__ == "__main__":
    menu()
