import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.signal import cwt

eeg_data = np.genfromtxt("test.csv")
eeg_data = eeg_data[:200,:]

# Simple moving average filter. The honda civic of all denoising techniques.
def sma(data, window_size):
    # Load EEG data into a numpy array
    eeg_data = np.array(data)

    # Apply moving average filter to EEG data
    filtered_data = np.convolve(eeg_data, np.ones(window_size)/window_size, mode='valid')

    # Return filtered data
    return filtered_data

# Kalman filtering, uses probability based state estimate techniques to disregard noise
def kalman(data):        
    # Read in the EEG data
    eeg_data = data

    # Initialize the Kalman filter
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                    observation_matrices=np.array([[1, 0]]),
                    initial_state_mean=np.array([0, 0]),
                    initial_state_covariance=np.eye(2),
                    transition_covariance=np.array([[0.0001, 0], [0, 0.0001]]),
                    observation_covariance=np.array([[0.1]]))

    # Perform the Kalman filter on the EEG data
    eeg_estimate, _ = kf.filter(eeg_data)
    return eeg_estimate


sma_filt = sma(eeg_data[:,1],3)
k_filt = kalman(eeg_data[:,1])[:,0]
print(np.shape(k_filt))
plt.plot(eeg_data[:,1], label = 'data')
plt.plot(sma_filt, label = 'sma')
plt.plot(k_filt, label = 'kalman')
plt.legend()
plt.show()

'''
# Not working correctly yet
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