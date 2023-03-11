import os
import time
from typing import Literal

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import argparse
import random

import numpy as np
import pandas as pd
import pygame
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

SENS = 150

# Bird physics
JUMP = 20
GRAVITY = 0.1
MAX_VEL = 5

# Pipe spawn rate
PIPE_INTERVAL = 200

# EEG receiver USB port
SERIAL_PORT = "/dev/ttyUSB0"

# Time (ms) to retroactively add marker to data
# motor signals take 20-30ms to travel
LABEL_WINDOW_MS = 40

FREQ = 250  # sample rate (hz)
SAMPLE_GAP_MS = 1000 / FREQ  # time between samples
LABEL_WINDOW = np.ceil(LABEL_WINDOW_MS / SAMPLE_GAP_MS).astype(int)


# Init game
pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((650, 1000))
font = pygame.font.Font("freesansbold.ttf", 32)
background = pygame.image.load("game/background.png")
background = pygame.transform.scale(
    background, [background.get_width() * 1.31, background.get_height() * 1.31]
)


def parseargs():
    parser = argparse.ArgumentParser()
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


def propagate_label(df, n):
    labels = df["marker"].copy().values
    for i in range(len(labels)):
        if labels[i] == 1:
            start = max(0, i - n)
            labels[start:i] = 1
    df["marker"] = labels
    return df


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
    df.index.name = "sample"

    df = propagate_label(df, LABEL_WINDOW)

    _, ext = os.path.splitext(fp)
    if not ext:
        fp += ".csv"

    df.to_csv(fp, index=True)


def menu():
    board = BoardShim(BoardIds.SYNTHETIC_BOARD, BrainFlowInputParams())
    fp = parseargs().file

    mode = None
    while True:
        if not mode:
            menu_text = [
                "Press space to play",
                "Press r to record",
                "Press b to play with bci",
                "Press q to quit",
            ]
        else:
            menu_text = [
                "Press space to continue",
                "Press q to quit",
            ]

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    if mode in ["record", "bci"]:
                        end_session(board, fp)
                    pygame.quit()
                    break

                if mode is None:
                    if event.key == pygame.K_SPACE:
                        mode = "normal"
                        run(mode, board)

                    elif event.key == pygame.K_r:
                        board = init_board()
                        mode = "record"
                        run(mode, board)

                    elif event.key == pygame.K_b:
                        board = init_board()
                        mode = "bci"
                        run(mode, board)

                # elif event.key == pygame.K_SPACE:
            if mode is not None:
                time.sleep(1)
                run(mode, board)

        update_screen(menu_text)


def update_screen(menu_text):
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
        hitbox_scale=[-55, -55],
        vel=0,
        img="game/brain.png",
    ) -> None:
        self.img = pygame.image.load(img)
        self.img = pygame.transform.scale(self.img, size)
        self.x = x
        self.y = y
        self.size = size
        self.hitbox_scale = hitbox_scale
        self.vel = vel

    def get_rect(self):
        return self.img.get_rect(
            center=(self.x + (self.size[0] // 2), self.y + (self.size[1] // 2))
        ).inflate(self.hitbox_scale)

    def jump(self):
        self.vel -= JUMP

    def update(self):
        self.vel += GRAVITY

        if abs(self.vel) > MAX_VEL:
            self.vel = MAX_VEL * (abs(self.vel) / self.vel)

        self.y += self.vel

    def draw(self):
        screen.blit(self.img, (self.x, self.y))
        # pygame.draw.rect(screen, (255, 0, 0), self.get_rect())


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


def run(mode: Literal["normal", "record", "bci"], board: BoardShim):
    bird = Bird()
    pipes = []
    bg_pos = 0

    end = False
    t = 0
    cooldown = 0
    while not end:
        t += 1
        screen.fill((255, 255, 255))

        clock.tick(150)

        # bci control (maybe poll as a pygame timer event?)
        if mode == "bci":
            data = board.get_current_board_data(60)

            c = [False, False]
            try:
                for i in range(1, 3):
                    if np.abs(data[i][55] - np.mean(data[i][0:50])) > SENS:
                        c[i - 1] = True
                if c[0] and c[1] and cooldown == 0:
                    cooldown = 50
                    bird.jump()
                elif cooldown > 0:
                    cooldown -= 1
            except:
                pass

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if mode != "bci" and event.key == pygame.K_SPACE:
                    if mode == "record":
                        board.insert_marker(1)
                    bird.jump()

        if t % PIPE_INTERVAL == 0:
            pipes.append(Pipe())

        # Checking Boundaries
        if bird.y <= -15 or bird.y >= 870:
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
