import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pywt

# Define labels for experiments
experiment_labels = [
    "1.5 MPA, 500 ms",#0
    "1.5 MPa, 500 ms",
    "2 MPa, 500 ms",
    "2 MPa, 750 ms",
    "2.5 MPa, 750 ms",
    "3 MPa, 750 ms",
    "3.5 MPa, 100 ms",
    "3.5 MPa, 200 ms",
    "2 MPa, 500 ms, new protocol",
    "2 MPa, 750 ms, new protocol",
    "2 MPa, 1000 ms, new protocol",
    "2 MPa, 1500 ms, new protocol",
    "2 MPa, 2000 ms, new protocol",
    "2.5 MPa, 2000 ms, new protocol",
    "2.5 MPa, 2500 ms, new protocol",#14
    "2.5 MPa, 3000 ms, new protocol",#15
    "3 MPa, 3000 ms, new protocol",
    "Change electrical", #17
    "Change electrical 1000 uA",
    "3.5 MPa, 3000 ms",#19
    "3.5 MPa, 3000 ms, 800 uA, 100 us",
    "3.5 MPa, 3000 ms, 800 uA, 200 us",
    "3.5 MPa, 3000 ms, 800 uA, 100 us",
    "3.5 MPa, 3000 ms, 600 uA, 100 us",
    "5 MPa, 100 ms, 600 uA, 100 us",
    "5 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",#25
    "4 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",
    "3 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",
    "4 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",#28
    "2 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",
    "1.5 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",#30 in fakerun
    "1 MPa, ES: 600 uA / 100 us, US: 500 rep / 5 ms",#31
    "3 MPa, ES: 600 uA / 100 us, US: 3000 ms",#32
    "4 MPa, ES: 600 uA / 100 us, US: 3000 ms " 
]

def filter_data(data, fs, fc=500, order=6, notch_freq=50, btype='low'):
    """
    Gets rid of the noise y applying band pass filter
 
    Returns:
        Data filtered of len 3: data_cuff, data_ta, data_gm
    """
    data_demeaned= data - np.mean(data)

    nyquist = 0.5 * fs

    b, a = signal.butter(order, fc / (0.5 * fs), btype, analog=False)
    filtered_data = signal.filtfilt(b, a, data_demeaned)

    # Design and apply notch filter for 50 Hz
    b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, Q=30)
    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)

    # Notch filters to remove frequencies between low_cut and high_cut
    notch_freqs = np.linspace(1, 30, num=30-1+1)
    for notch_freq in notch_freqs:
        notch = notch_freq / nyquist
        b_notch, a_notch = signal.iirnotch(notch, 30)
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)
    
    return filtered_data


def segment_data(array, n_parts):
    """
    Segments into trials

    Parameters:
        n_parts (int): number of trials for the experiment

    Returns:
        2D array of shape (3, n_parts) where 3 is for cuff, ta and gm
    """
    # Calculer la longueur de chaque segment
    segment_length = len(array) // n_parts

    # Créer une liste pour stocker les segments
    segments = []

    for i in range(n_parts):
        start_index = i * segment_length
        # Pour le dernier segment, prendre tout jusqu'à la fin de l'array
        if i == n_parts - 1:
            segments.append(array[start_index:])
        else:
            segments.append(array[start_index:start_index + segment_length])

    return segments

def calculate_scale(data, fs):
    """
    Calculate the time scale for the given data based on the sampling frequency.

    Parameters:
        data (list): 2D array representing the data where each row corresponds to a trial.
        fs (float): Sampling frequency, i.e., the number of samples per second.

    Returns:
        numpy.ndarray: Time scale array corresponding to the data, scaled according to the sampling frequency.
    """
    #timescale
    x_axis_trial = np.arange(len(data[0]))
    return x_axis_trial / fs

def chose_protocol(number):
    """
    Define the design of the experiment. Can add protocols as you wish

    Parameters:
        number (int): if first or second protocol

    Returns:
        tuple: A tuple containing the following elements:
            - list: ES (electrical stimulation) time points
            - float: US (ultrasound) start time
            - float: US duration
    """

    if number==1:
        #design experiment in seconds 
        US_length= 3 #duration of sonication
        ES_first=0.1 #beginning of ES stimulation
        ES_duration = 0.0002 
        ES_stimint_1= 0.1+ES_duration #interval between stimulation for each burst
        
        # First ES stage
        firstES = [ES_first + i * ES_stimint_1 for i in range(5)] #modify the 5 if more bursts

        ES_second= firstES[4]+0.42 #ES after break 
        US_start=ES_second
        ES_stimint_2= 0.2+ES_duration

        # Second ES stage
        secondES = [ES_second + i * ES_stimint_2 for i in range(10)]

        #build entire trial
        ES = firstES + secondES

        return ES, US_start, US_length
    
    if number==2:
        #design experiment in seconds 
        US_length= 1 #duration of sonication
        ES_first=0.1 #beginning of ES stimulation
        ES_duration=0.0002

        ES_stimint_1= 0.1+ES_duration#interval between stimulation for each burst
        # First ES stage
        firstES = [ES_first + i * ES_stimint_1 for i in range(5)]

        ES_second= firstES[4]+0.4#ES after break 
        US_start=ES_second
        ES_stimint_2= 0.2+ES_duration
        ES_stimint_3= 0.1+ES_duration

        # Second ES stage
        secondES = [ES_second + i * ES_stimint_2 for i in range(5)]

        #Third ES stage
        ES_third = secondES[4] + 0.4 #ES after 2nd break
        thirdES = [ES_third + i * ES_stimint_3 for i in range(5)]

        #make big list
        ES= firstES+secondES+thirdES
        
        return ES, US_start, US_length
    
def get_number_parameters(experiment_parameters):
    """
    Extracts numerical values from a string containing experiment parameters. Useful for the plots' parameters in the result viewing

    Args:
        experiment_parameters (str): A string containing experiment parameters.

    Returns:
        List[Union[int, float]]: A list of numeric values extracted from the input string.

    Example:
        >>> experiment_params = "temperature 25.5 pressure 1013.2 duration 60"
        >>> numbers = get_number_parameters(experiment_params)
        >>> print(numbers)
        [25.5, 1013.2, 60]
    """
    numbers= [float(s) if '.' in s else int(s) for s in experiment_parameters.split() if s.replace('.', '').isdigit()]
    return numbers

def compare_parameters(param1, param2):
    """
    Compare parameters of two experiments. To know which parameters are varying and which parameter is constant across comparison

    Parameters:
        param1 (np.array): Parameters of the first experiment.
        param2 (np.array): Parameters of the second experiment.

    Returns:
        str: A message indicating which parameter differs ('pressure', 'duration', or 'both').
    """
    
    if param1[0] != param2[0]:
        return "Pressure differs.", [param1[0], param2[0]]
    elif param1[1] != param2[1]:
        return "Duration differs.", [param1[1], param2[1]]
    
