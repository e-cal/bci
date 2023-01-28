import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import cwt
from scipy.signal import spectrogram

eeg_data = np.genfromtxt("test.csv")

eeg_data = eeg_data[:200, :]
# Simple moving average filter. The honda civic of all denoising techniques.
def sma(data, window_size):
    # Load EEG data into a numpy array
    eeg_data = np.array(data)

    # Apply moving average filter to EEG data
    filtered_data = np.convolve(
        eeg_data, np.ones(window_size) / window_size, mode="valid"
    )

    # Return filtered data
    return filtered_data


def bandpass(data, lowcut, highcut, fs):
    # Define the sample rate of the data
    fs = 250  # Hz

    # Create the bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# Kalman filtering, uses probability based state estimate techniques to disregard noise
def kalman(data):
    # Read in the EEG data
    eeg_data = data

    # Initialize the Kalman filter
    kf = KalmanFilter(
        transition_matrices=np.array([[1, 1], [0, 1]]),
        observation_matrices=np.array([[1, 0]]),
        initial_state_mean=np.array([0, 0]),
        initial_state_covariance=np.eye(2),
        transition_covariance=np.array([[0.0001, 0], [0, 0.0001]]),
        observation_covariance=np.array([[0.1]]),
    )

    # Perform the Kalman filter on the EEG data
    eeg_estimate, _ = kf.filter(eeg_data)
    return eeg_estimate


"""
sma_filt = sma(eeg_data[:,1],3)
k_filt = kalman(eeg_data[:,1])[:,0]
print(np.shape(k_filt))
plt.plot(eeg_data[:,1], label = 'data')
plt.plot(sma_filt, label = 'sma')
plt.plot(k_filt, label = 'kalman')
plt.legend()
plt.show()
"""


def spec(data):
    """
    # Define the sampling rate (in Hz) and window size (in seconds) for the spectrogram
    fs = 250

    # Calculate the spectrogram
    f, t, Sxx = spectrogram(data, fs)

    # Plot the spectrogram
    plt.pcolormesh(t, f, Sxx)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    import numpy as np
    from scipy import signal
    from scipy.fft import fftshift
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    fs = 10
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500*np.cos(2*np.pi*0.25*time)
    carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise
    f, t, Sxx = signal.spectrogram(data, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    """


"""
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

"""
