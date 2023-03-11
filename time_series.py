from classifier import *


file_path = os.path.join(
        r"C:\Users\danie\Documents\GitHub\bci\data", "3-games.csv"
    )

data = pd.read_csv(file_path, sep=",", header=0)
data = get_eeg(data)
data = data[:][0:1200]
print(data)

for i in range(len(data['eeg1'])):
    if data['marker'][i] == 1:
        plt.subplot(1,2,1)
        plt.plot(data['eeg1'][i-50:i+50])
        plt.axvline(i, color = "yellow")
        plt.subplot(1,2,2)
        plt.axvline(i, color = "yellow")
        plt.plot(data['eeg2'][i-50:i+50])
        plt.show()

