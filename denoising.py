import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.signal import cwt


eeg_data = np.genfromtxt("test.csv")

print(np.shape(eeg_data))
plt.plot(eeg_data[:,1])


# Simple moving average filter. The honda civic of all denoising techniques.
def sma(data, window_size):
    # Load EEG data into a numpy array
    eeg_data = np.array(data)

    # Apply moving average filter to EEG data
    filtered_data = np.convolve(eeg_data, np.ones(window_size)/window_size, mode='valid')

    # Return filtered data
    return filtered_data

filtered = sma(eeg_data[:,1],3)
plt.plot(filtered)
plt.show()
'''

# Not working correctly yet
def kalman(data, transition_matrices, observation_matrices, initial_state_mean,initial_state_covariance,observation_covariance,transition_covariance):
    # Define the Kalman filter
    kf = KalmanFilter(transition_matrices,
                    observation_matrices,
                    initial_state_mean,
                    initial_state_covariance,
                    observation_covariance,
                    transition_covariance)

    # Apply the Kalman filter to the EEG data
    eeg_filtered, _ = kf.filter(data)
    return eeg_filtered

def cwt(data):

    # Load the noisy signal
    noisy_signal = data

    # Define the wavelet to use
    wavelet = 'cmor1.5-1.0'

    # Step 1: Apply the continuous wavelet transform
    scales, coefs, _ = cwt(noisy_signal, wavelet, 1)

    # Step 2: Threshold the coefficients
    # Define the threshold value
    threshold = np.percentile(np.abs(coefs), 95)
    # Set the coefficients below the threshold to zero
    coefs[np.abs(coefs) < threshold] = 0

    # Step 3: Reconstruct the denoised signal
    # Apply the inverse continuous wavelet transform
    denoised_signal = icwt(coefs, wavelet, 1, scales)

    # Step 4: Additional processing

    # Plot the raw and denoised signals
    import matplotlib.pyplot as plt
    plt.plot(noisy_signal, label='Noisy signal')
    plt.plot(denoised_signal, label='Denoised signal')
    plt.legend()
    plt.show()

'''