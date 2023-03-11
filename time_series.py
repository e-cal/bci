from classifier import *
import torch
import torch.nn as nn

file_path = os.path.join(
        r"C:\Users\danie\Documents\GitHub\bci\data", "blink.csv"
    )

data = pd.read_csv(file_path, sep=",", header=0)
data = get_eeg(data)

buffer = 150

for i in range(len(data['eeg1'])):
    if data['marker'][i] == 1:
        plt.subplot(1,2,1)
        plt.plot(data['eeg1'][i-buffer:i+buffer])
        plt.axvline(i, color = "yellow")
        plt.subplot(1,2,2)
        plt.axvline(i, color = "yellow")
        plt.plot(data['eeg2'][i-buffer:i+buffer])
        plt.show()

# Building training datastructure from blinking eeg data

'''def parse_blinking_data(data, window_size, buffer):
    train = pd.DataFrame(columns=range(window_size*2+1))
    for i in range(len(data['eeg1'])):
        if data['marker'][i] == 1:
            # if there is a blink, grab time series data from channel 1 and 2
            channel1 = np.array(data['eeg1'][i-window_size][i])
            channel2 = np.array(data['eeg2'][i-window_size][i])
            marker = np.array(data['marker'][i])
            
            # for every blink, grab 2 non-blinking data points, random distances form the blink within 1-2s
            
            train.append(np.concatenate(channel1, channel2, marker))
            
            channel1_neg= np.array(data['eeg1'][i-window_size-buffer][i])
            channel2_neg= np.array(data['eeg2'][i-window_size-buffer][i])
            marker_neg = 0'''
            
            

'''class Net(nn.Module):
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

'''

