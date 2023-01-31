import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import argparse
import random

import numpy as np
import pandas as pd
import pygame
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

JUMP = 20
GRAVITY = 0.1

MAX_VEL = 5

SERIAL_PORT = "/dev/ttyUSB0"


# Init game
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((650, 1000))
font = pygame.font.Font("freesansbold.ttf", 32)
background = pygame.image.load("game/background.png")
background = pygame.transform.scale(
    background, [background.get_width() * 1.31, background.get_height() * 1.31]
)
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--record",
        help="record EEG data",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--bci",
        help="control using BCI",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="file path to save data to (default: data.csv)",
        required=False,
        default="data.csv",
    )
    return parser.parse_args()


def init_board() -> BoardShim:
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board_id = BoardIds.CYTON_BOARD
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    return board


def end_session(board: BoardShim, fp: str):
    data = board.get_board_data()

    board.stop_stream()
    board.release_session()

    df = pd.DataFrame(np.transpose(data))
    df.columns = [
        "packet",
        "eeg1",
        "eeg2",
        "eeg3",
        "eeg4",
        "eeg5",
        "eeg6",
        "eeg7",
        "eeg8",
        "accel1",
        "accel2",
        "accel3",
        "other1",
        "other2",
        "other3",
        "other4",
        "other5",
        "other6",
        "other7",
        "analog1",
        "analog2",
        "analog3",
        "timestamp",
        "marker",
    ]

    # preprocess here maybe?

    df.to_csv(fp, index=True)


def menu():
    menu_text = [
        "Press space to play",
        "Press q to quit",
    ]

    args = parseargs()

    if args.record or args.bci:
        if args.record and args.bci:
            print("Cannot record and control with BCI at the same time.")
            exit()

        eeg = True
        board = init_board()

    else:  # not using eeg. fake a board for type checking
        eeg = False
        board = BoardShim(BoardIds.SYNTHETIC_BOARD, BrainFlowInputParams())

    needs_update = True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                needs_update = True
                if event.key == pygame.K_q:
                    if eeg:
                        end_session(board, args.file)
                    pygame.quit()
                    break

                elif event.key == pygame.K_SPACE:
                    run(args.record, board)

        if needs_update:
            needs_update = False
            try:
                screen.blit(background, (0, 0))
                for i, t in enumerate(menu_text):
                    text = font.render(t, True, (0, 0, 0), None)
                    text_rect = text.get_rect()
                    text_rect.center = (
                        screen.get_width() // 2,
                        (screen.get_height() // 2) - (len(menu_text) * 20) + (i * 40),
                    )
                    screen.blit(text, text_rect)
                pygame.display.update()

            except pygame.error:
                exit()


class Bird:
    def __init__(
        self,
        x=100,
        y=screen.get_height() // 2,
        size=[90, 90],
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
        self.img = pygame.transform.scale(
            self.img, [self.img.get_width() * 1.05, self.img.get_height() * 1.1]
        )
        self.imginv = pygame.transform.flip(self.img, False, True)

        self.x = screen.get_width() + 50
        self.mid = random.randint(300, 700)
        # self.mid = random.choice([300, 800])
        self.gap = random.randint(250, 300)
        self.update()

    def collide(self, bird):
        return bird.colliderect(self.top) or bird.colliderect(self.bottom)

    def update(self):
        self.x -= 2
        self.top = self.img.get_rect(midbottom=(self.x, self.mid - (self.gap // 2)))
        self.bottom = self.img.get_rect(midtop=(self.x, self.mid + (self.gap // 2)))

    def draw(self):
        screen.blit(self.img, self.bottom)
        screen.blit(self.imginv, self.top)


def update_background(pos):
    pos -= 1
    screen.blit(background, (pos, 0))
    screen.blit(background, (pos + 650, 0))

    if pos < -650:
        pos = 0

    return pos


def run(record: bool, board: BoardShim):
    bird = Bird()
    pipes = []
    bg_pos = 0

    end = False
    while not end:
        screen.fill((255, 255, 255))

        clock.tick(150)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    if record:
                        print("jump")
                        board.insert_marker(1)
                    bird.jump()

            if event.type == SPAWNPIPE:
                pipes.append(Pipe())

        # Checking Boundaries
        if bird.y >= screen.get_height() or bird.y <= -40:
            end = True

        bg_pos = update_background(bg_pos)

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
