from classifier import *
import torch
import torch.nn as nn


GAIN = 24
SCALE = 450000/GAIN/(2**23 - 1)
REACTION_TIME = 150
BLINK_DURATION = 125

def reaction_adjust(df, n):
    for i in range(len(df["marker"])-n):
        if df["marker"][i] == 1.0:
            df["marker"][i] = 0
            df["marker"][i+n] = 1
    return df

# Building training datastructure from blinking eeg data

file_path = os.path.join(
        r"C:\Users\danie\Documents\GitHub\bci\data", "blink.csv"
    )

data = pd.read_csv(file_path, sep=",", header=0)
data = get_eeg(data)

buffer = 250

'''for i in range(len(data['eeg1'])):
    if data['marker'][i] == 1:
        plt.subplot(1,2,1)
        plt.plot(data['eeg1'][i-buffer + REACTION_TIME:i+buffer + REACTION_TIME])
        plt.axvline(i + REACTION_TIME, color = "yellow")

        plt.axvline(i+REACTION_TIME, color = "red")
        
        plt.subplot(1,2,2)
        plt.axvline(i + REACTION_TIME, color = "yellow")
        plt.plot(data['eeg2'][i-buffer:i+buffer])

        plt.axvline(i+REACTION_TIME, color = "red")
        plt.show()'''



def parse_blinking_data(data, window_size, buffer):
    train = pd.DataFrame(columns=range(window_size*2+1))
    for i in range(len(data['eeg1'])):
        if data['marker'][i] == 3:
            # if there is a blink, grab time series data from channel 1 and 2
            channel1 = np.array(data['eeg1'][i - window_size:i])
            channel2 = np.array(data['eeg2'][i - window_size:i])
            marker = np.array([1])

            datum = np.concatenate((channel1, channel2, marker))
            datum = np.reshape(datum,(1,251))

            df = pd.DataFrame(datum)
            train = pd.concat([train, df])

            # for every blink, grab a non-blinking data sample,from before the blink
            channel1_neg= np.array(data['eeg1'][i - window_size - buffer:i - buffer])
            channel2_neg= np.array(data['eeg2'][i - window_size - buffer:i - buffer])
            marker_neg = np.array([0])

            datum = np.concatenate((channel1_neg, channel2_neg, marker_neg))
            datum = np.reshape(datum,(1,251))

            df = pd.DataFrame(datum)
            train = pd.concat([train, df])
        return train

train = parse_blinking_data(data, BLINK_DURATION, buffer)
print(train)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(50, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        self.fc3 = nn.Linear(2,1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = Net()


