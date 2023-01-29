import time
import argparse
import pandas as pd
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--eeg",
        type=bool,
        help="Get EEG electrode readings only",
        required=False,
        default=False,
    )
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

    eeg_channels = BoardShim.get_eeg_channels(board_id.value)
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

    print("Data From the Board")
    print(df.head())

    df.to_csv(args.file, index=False)
    # DataFilter.write_file(data, args.file, "w")
    # print(f"Data saved to {args.file}")

    """
    restored_data = DataFilter.read_file(args.file)
    restored_df = pd.DataFrame(np.transpose(restored_data))  # type: ignore
    print("Data From the File")
    print(restored_df.head(10))
    """


if __name__ == "__main__":
    main()
