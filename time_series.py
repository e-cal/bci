from preprocessing import *
from classifier import *
import torch
import torch.nn as nn


GAIN = 24
SCALE = 450000 / GAIN / (2**23 - 1)
BLINK_DURATION = 100
REACTION_TIME = 110


def reaction_adjust(df, n):
    for i in range(len(df["marker"]) - n):
        if df["marker"][i] == 1.0:
            df["marker"][i] = 0
            df["marker"][i + n] = 1
    return df


# Building training datastructure from blinking eeg data

file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\bci\data", "blink.csv")

data = pd.read_csv(file_path, sep=",", header=0)
data = get_eeg(data)

buffer = 250

"""for i in range(len(data['eeg1'])):
    if data['marker'][i] == 1:
        plt.subplot(1,2,1)
        plt.plot(data['eeg1'][i-buffer + REACTION_TIME:i+buffer + REACTION_TIME])
        plt.axvline(i + REACTION_TIME, color = "yellow")

        plt.axvline(i+REACTION_TIME, color = "red")
        
        plt.subplot(1,2,2)
        plt.axvline(i + REACTION_TIME, color = "yellow")
        plt.plot(data['eeg2'][i-buffer:i+buffer])

        plt.axvline(i+REACTION_TIME, color = "red")
        plt.show()"""

print(data)

def parse_blinking_data(data, window_size, buffer):
    train = pd.DataFrame(columns=range(window_size * 2 + 1))
    for i in range(len(data["eeg1"])):
        if data["marker"][i] == 1:
            # if there is a blink, grab time series data from channel 1 and 2
            channel1 = np.array(
                data["eeg1"][i - window_size + REACTION_TIME : i + REACTION_TIME + BLINK_DURATION]
            )
            # channel1 = bandpass(channel1, 1, 30)
            channel2 = np.array(
                data["eeg2"][i - window_size + REACTION_TIME : i + REACTION_TIME + BLINK_DURATION]
            )
            # channel2 = bandpass(channel2, 1, 30)
            
            
            marker = np.array([1])

            datum = np.concatenate((channel1, channel2, marker))
            datum = np.reshape(datum, (1, len(channel1) + len(channel2) + 1))
            
            if False:
                plt.subplot(1,2,1)
                plt.plot(channel1)
                plt.subplot(1,2,2)
                plt.plot(sma(data['eeg1'][i - window_size + REACTION_TIME : i + REACTION_TIME + BLINK_DURATION],30))
                plt.show()
            

            df = pd.DataFrame(datum)
            train = pd.concat([train, df])

            # for every blink, grab a non-blinking data sample,from before the blink
            channel1_neg = np.array(data["eeg1"][i - window_size + REACTION_TIME : i + REACTION_TIME + BLINK_DURATION])
            channel2_neg = np.array(data["eeg2"][i - window_size + REACTION_TIME : i + REACTION_TIME + BLINK_DURATION])
            marker_neg = np.array([0])

            datum = np.concatenate((channel1_neg, channel2_neg, marker_neg))
            datum = np.reshape(datum, (1, len(channel1) + len(channel2) + 1))

            df = pd.DataFrame(datum)
            train = pd.concat([train, df])
    return train


train = parse_blinking_data(data, 100, buffer)


save = True

if save:
    train.to_csv('blink_1', index=False)

