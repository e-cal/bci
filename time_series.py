from preprocessing import *
from classifier import *
import torch
import torch.nn as nn


GAIN = 24
SCALE = 450000 / GAIN / (2**23 - 1)
BLINK_DURATION = 100
REACTION_TIME = 110

# Building training datastructure from blinking eeg data

file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\bci\data", "blink-3m.csv")

data = pd.read_csv(file_path, sep=",", header=0)
data = get_eeg(data)

buffer = 350

'''for i in range(len(data['eeg1'])):
    if data['marker'][i] == 1:
        plt.subplot(1,2,1)
        plt.plot(data['eeg1'][i: i + 176])
        #plt.axvline(i, color = "yellow")

        plt.subplot(1,2,2)
        plt.plot(data['eeg1'][i - 176 : i])
        plt.show()'''

def parse_blinking_data(data, window_size, buffer, smooth=bool):
    train = pd.DataFrame(columns=range(176-30))
    eye_channels = ['eeg1']
    for channel in eye_channels:
        # do the following for each eeg channel above the eye
        for i in range(len(data["eeg1"])):
            if data["marker"][i] == 1:
                # if there is a blink, grab time series data from channel
                blink_data = np.array(
                    data[channel][i : i +176]
                )

                # Find the index of the minimum value
                min_index = np.argmin(blink_data)
                
                # Average the data
                average = sma(data[channel][i + min_index - 88: i + min_index + 88], 30)
                marker = np.array([1])

                # concatenate the time series data with a marker at the end
                datum = np.concatenate((average, marker))
                if len(datum) > 100:
                    datum = np.reshape(datum, (1, len(datum)))
                    df = pd.DataFrame(datum)
                    train = pd.concat([train, df])

                # for every blink, grab a non-blinking data sample,from before the blink
                nonblink_data = np.array(data[channel][i - 176 : i])

                # average this data as well
                average = sma(data[channel][i - 176 : i], 30)

                marker_neg = np.array([0])

                # concatenate the time series data with a marker at the end
                datum = np.concatenate((average, marker_neg))
                if len(datum) > 100:
                    datum = np.reshape(datum, (1, len(datum)))
                    df = pd.DataFrame(datum)
                    train = pd.concat([train, df])
    return train

train = parse_blinking_data(data, 100, buffer)

save = True

if save:
    train.to_csv('blink_1_averaged.csv', index=False)

