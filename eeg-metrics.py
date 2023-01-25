import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import (
    MLModel,
    BrainFlowMetrics,
    BrainFlowClassifiers,
    BrainFlowModelParams,
)


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
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    args = parseargs()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    board_id = BoardIds.CYTON_BOARD

    board = BoardShim(board_id, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    print(f"sampling rate: {sampling_rate}")
    board.prepare_session()
    board.start_stream(45000)
    BoardShim.log_message(
        LogLevels.LEVEL_INFO.value, "start sleeping in the main thread"
    )
    time.sleep(
        5
    )  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    feature_vector = bands[0]
    print(feature_vector)

    mindfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
    )
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    print("Mindfulness: %s" % str(mindfulness.predict(feature_vector)))
    mindfulness.release()

    restfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.RESTFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
    )
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    print("Restfulness: %s" % str(restfulness.predict(feature_vector)))
    restfulness.release()


if __name__ == "__main__":
    main()
