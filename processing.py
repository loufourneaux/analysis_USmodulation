import numpy as np
from scipy.integrate import trapz
from preprocessing import *
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def determine_es_samples(es_values, fs, w=0.1, o=0):
    """
    Determine the electrical stimulations for the design experiment.

    Args:
        es_intervals (list): A list of electrical stimulation intervals (ES) in seconds.
        fs (float): The sampling rate (in Hz) of the experiment.
        w (float, optional): Width of the stimulation interval (default is 0.1 seconds).
        o (float, optional): Offset for adjusting the ES intervals (default is 0 seconds).

    Returns:
        list: A list of ES samples (rounded to the nearest integer) corresponding to the input intervals. 
    """
    # Create stimulation intervals in time domain around each ES
    stim_intervals = []
    for es_value in es_values:
        stim_interval = [es_value - w * 0.2 + o, es_value + w * 0.2 + o] #0.1 is the ESinterval the smallest possible
        stim_intervals.append(stim_interval)
    
    #convert intervals to sampling rate domain
    es_samples = []
    for interval in stim_intervals:
        es_sample = [round(i * fs) for i in interval]
        es_samples.append(es_sample)

    return es_samples

def extract_stimulation_segments(raw_data, es_samples):
    """
    Extracts stimulation segments from raw data based on specified ES samples.

    Args:
        raw_data (list or numpy array): The raw data containing the entire recording.
        es_samples (list of tuples): A list of tuples representing ES sample intervals (start, end).

    Returns:
        list: A list of stimulation segments extracted from the raw data.
    """

    stim_segments = []
    for start, end in es_samples:
        segment = raw_data[start:end]
        stim_segments.append(segment)
    return stim_segments

def extract_action_potentials(raw_data, es_values, fs, w=0.1, o=0, p=4):
    """
    Extract action potentials from raw recordings.

    Args:
        raw_data (numpy.ndarray): Raw voltage recordings.
        es_intervals (list): List of timestamps corresponding to electrical stimulation.
        fs (float): Sampling frequency in Hz.
        w (float, optional): Window size for action potential extraction (default is 0.1 seconds).
        o (int, optional): Offset in samples (default is 0).
        p (int, optional): Polynomial order for baseline subtraction (default is 4).

    Returns:
        List of action potentials for one trial (size 15 if 15 ES).
    """
    # Determine es_samples
    es_samples= determine_es_samples(es_values, fs, w=w, o=o)
    
    # Extract stimulation segments for each trial
    stim_segments = extract_stimulation_segments(raw_data, es_samples)
    
    return stim_segments


def area_under_curve(action_potentials, fs):
    """
    Calculate the area under the curve for action potentials.

    Args:
        action_potentials (list): List of action potentials.
        fs (float): Sampling frequency in Hz.

    Returns:
        list: List of areas under the curve for each action potential.
    """
    x_axis= calculate_scale(action_potentials, fs)
    areas = [trapz(ap, x_axis) for ap in action_potentials]

    return areas


def normalize_action_potentials(action_potentials):
    """
    Normalize action potentials by dividing each value by the maximum amplitude.

    Args:
        action_potentials (list): List of action potentials.

    Returns:
        list: List of normalized action potentials.
    """
    normalized_action_potentials = []
    for ap in action_potentials:
        max_amplitude = max(ap)
        normalized_ap = [v / max_amplitude for v in ap]
        normalized_action_potentials.append(normalized_ap)
    return normalized_action_potentials

    
def find_thresholds(action_potentials, x_axis):
    """
    Find the threshold of each action potential by taking the maximum of the derivative.
    
    Parameters:
        action_potentials (numpy.ndarray): A 1D array where each row represents an individual action potential.
        x_axis (numpy.ndarray): The x-axis values corresponding to the action potentials.
    
    Returns:
        numpy.ndarray, numpy.ndarray: Arrays containing the threshold value and its corresponding x-axis value for each action potential.
    """
    thresholds = []
    latencies = []
    for ap in action_potentials:
        # Find the maximum amplitude of the normalized action potential
        max_amplitude = max(ap)
        
        # Calculate the threshold as 30 percent of the maximum amplitude
        threshold = 0.4 * max_amplitude
        
        # Find the index where the curve crosses the threshold
        threshold_index = np.argmax(np.array(ap) > threshold)
        
        # Get the corresponding x-axis value
        latency = x_axis[threshold_index]
        
        thresholds.append(threshold)
        latencies.append(latency)
        
    return thresholds, latencies

def find_latencies(action_potentials, x_axis):
    """
    Calculate latencies for action potentials.

    Args:
        action_potentials (list): List of action potentials.
        x_axis (numpy.ndarray): Time axis corresponding to the action potentials.

    Returns:
        tuple: A tuple containing:
            - thresholds (list): List of threshold values.
            - thresholds_x_values (list): List of corresponding x-axis values for the thresholds.
            - latencies (numpy.ndarray): Latency values calculated as thresholds_x_values minus the midpoint of x_axis.
    """
    thresholds, thresholds_x_values = find_thresholds(action_potentials, x_axis)
    latencies = thresholds_x_values - ((max(x_axis)-min(x_axis))/2)
    return thresholds, thresholds_x_values, latencies 


def find_last_crossing_time(action_potentials, x_axis):
    """
    Find the last crossing time of a certain threshold for each action potential.

    Parameters:
        action_potentials (numpy.ndarray): A 2D array where each row represents an individual action potential.
        x_axis (numpy.ndarray): The x-axis values corresponding to the action potentials.
        threshold_ratio (float): The ratio of the threshold to the maximum amplitude.

    Returns:
        numpy.ndarray: Array containing the last crossing time for each action potential.
    """
    last_crossing_times = []
    for ap in action_potentials:
        # Find the maximum amplitude of the normalized action potential
        max_amplitude = max(ap)
        
        # Calculate the threshold
        threshold = 0.1 * max_amplitude #change percentage as you wish
        
        # Find the indices where the curve crosses the threshold
        crossing_indices = np.where(np.array(ap) > threshold)[0]
        
        # If no crossings found, append None
        if len(crossing_indices) == 0:
            last_crossing_times.append(None)
        else:
            # Get the last crossing index and corresponding x-axis value
            last_crossing_index = crossing_indices[-1]
            last_crossing_time = x_axis[last_crossing_index]
            last_crossing_times.append(last_crossing_time)
        
    return np.array(last_crossing_times)


def calculate_peak_to_peak_distances(action_potentials):
    """
    Calculate the peak-to-peak distances of each action potential.
    
    Parameters:
        action_potentials (list of numpy.ndarray): List of action potentials.
    
    Returns:
        list of float: Peak-to-peak distances for each action potential.
    """
    peak_to_peak_distances = []
    for i, ap in enumerate(action_potentials):
        # Find peaks and troughs
        peaks, _ = find_peaks(ap)
        troughs, _ = find_peaks(-ap)  # Invert action potential to find troughs
        
        # Ensure at least one peak and trough are found
        if len(peaks) > 0 and len(troughs) > 0:
            # Calculate peak-to-peak distance as the difference between the maximum peak and minimum trough
            peak_to_peak_distance = ap[peaks].max() - ap[troughs].min()
            peak_to_peak_distances.append(peak_to_peak_distance)
        else:
            peak_to_peak_distances.append(None)  # Handle cases where peaks or troughs are not found
    
    return peak_to_peak_distances


def fourier_transform(signal, sampling_rate):
    """
    Apply Fourier transfrom to the recordings
    """

    # FFT
    N = len(signal)
    T = 1.0 / sampling_rate
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    yf_positive = 2.0/N * np.abs(yf[:N//2])

    return xf, yf_positive

def remove_outliers(data):
    """
    Remove outliers from a list of data using the IQR method.
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def remove_outliers_from_dict(data_dict):
    """
    Remove outliers from each list in a dictionary using the IQR method.
    
    Args:
        data_dict (dict): Dictionary containing lists of data.
        
    Returns:
        dict: Dictionary with outliers removed from each list.
    """

    return {key: [remove_outliers(sublist) for sublist in value] for key, value in data_dict.items()}


# Function to calculate FFT power
def calc_fft_power(EMG_windows, fs):
    N = EMG_windows.shape[1]  # Number of points in each window
    freqs = np.fft.rfftfreq(N, 1/fs)  # Frequency bins
    fft_vals = np.fft.rfft(EMG_windows, axis=1)
    fft_power = np.abs(fft_vals) ** 2  # Power spectrum
    return freqs[1:], fft_power[:, 1:]

# Function to extract features
def extract_features(EMG_windows, fs):
    mav = np.mean(np.absolute(EMG_windows), axis=1)
    maxav = np.max(np.absolute(EMG_windows), axis=1)
    std = np.std(EMG_windows, axis=1)
    rms = np.sqrt(np.mean(np.square(EMG_windows), axis=1))
    wl = np.sum(np.abs(np.diff(EMG_windows, axis=1)), axis=1)
    zc = np.sum(np.diff(np.sign(EMG_windows), axis=1) != 0, axis=1)
    diff = np.diff(EMG_windows, axis=1)
    ssc = np.sum((diff[:, :-1, :] * diff[:, 1:, :]) < 0, axis=1)

    freqs, fft_power = calc_fft_power(EMG_windows, fs=fs)
    mean_power = np.mean(fft_power, axis=1)
    tot_power = np.sum(fft_power, axis=1)
    freqs_reshaped = freqs.reshape(1, freqs.shape[0], 1)
    mean_frequency = np.sum(fft_power * freqs_reshaped, axis=1) / np.sum(fft_power, axis=1)
    cumulative_power = np.cumsum(fft_power, axis=1)
    total_power = cumulative_power[:, -1, :]
    median_frequency = np.zeros((EMG_windows.shape[0], EMG_windows.shape[2]))

    for i in range(EMG_windows.shape[0]):
        for j in range(EMG_windows.shape[2]):
            median_frequency[i, j] = freqs[np.where(cumulative_power[i, :, j] >= total_power[i, j] / 2)[0][0]]

    peak_frequency = freqs[np.argmax(fft_power, axis=1)]

    X = np.column_stack((mav, maxav, std, rms, wl, zc, ssc, mean_power, tot_power, mean_frequency, median_frequency, peak_frequency))

    return X
  










