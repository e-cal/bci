import argparse
import os
import time

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--serial-port",
        type=str,
        help="serial port for reciever",
        required=False,
        default="/dev/ttyUSB0",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="file path to save data to",
        required=False,
        default="data.csv",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        help="Length of time (seconds) to read data for",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        help="Label to add to the marker col of data",
        required=False,
        default=0,
    )
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    board_id = BoardIds.CYTON_BOARD

    board = BoardShim(board_id, params)
    board.prepare_session()

    board.start_stream()
    time.sleep(args.time)

    data = board.get_board_data()  # get all data and remove it from internal buffer
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

    df["marker"] = args.label

    fp = args.file
    _, ext = os.path.splitext(fp)
    if not ext:
        fp += ".csv"

    df.to_csv(fp, index=True)


if __name__ == "__main__":
    main()
