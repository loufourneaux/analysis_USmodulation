import matplotlib.pyplot as plt
import numpy as np
from preprocessing import *
from processing import *
from scipy.fft import fft, fftfreq

def plot_experiment_overview(data, experiment_number, parameter, n_parts, fs):
    """
    Plot the experiment overview: number of trials and recordings 

    Args:
        data (list of lists): Raw data for each part of the experiment (e.g., CUFF, TA, GM).
        experiment_number (int): The experiment number.
        parameter (str): Description of the experimental parameter.
        n_parts (int): Number of parts in the experiment.
        fs (float): Sampling frequency in Hz.

    Returns:
        None
    """
    #timescale
    
    x_axis_trial_s = calculate_scale(data[0], fs)

    #figure
    fig, axes = plt.subplots(3, n_parts, figsize=(8, 10), sharey='row')
    for row, data in enumerate(data):
        for col in range(n_parts):
            axes[row][col].plot(x_axis_trial_s, data[col])
            axes[row][col].xaxis.grid(True)
            if col > 0:
                axes[row][col].tick_params(axis='y', which='both', left=False)
    for i, titre in enumerate(["CUFF", "TA", "GM"]):
        axes[i][0].set_ylabel(titre, weight='bold', fontsize='large', rotation=90, labelpad=15)
        axes[i][0].yaxis.set_label_position("left")
    for i in range(n_parts):
        axes[0][i].set_title(f"Trial {i+1}")
    for ax in axes[0:-1, :].flatten():
        ax.set_xticklabels([])
    axes[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(
        f"Overview of experiment {experiment_number+1}, parameters: {parameter} ", fontsize='large', weight='bold')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1, hspace=0.1, wspace=0.1)
    #plt.show()
    plt.savefig(f"figures/experiment_overview/exp_{experiment_number+1}.png")
    plt.close()


def plot_design_experiment(data, experiment_number, experiment_param, US_start, US_length, ES, fs ):
    """
    Visualize the experiment versus the design to check if they match.

    Args:
        data (list of lists): Raw data for each part of the experiment (e.g., CUFF, TA, GM).
        experiment_param (str): Description of the experimental parameter.
        US_start (float): Start time of the ultrasound (US) stimulus in seconds.
        US_length (float): Duration of the US stimulus in seconds.
        ES (float): Time of the electrical stimulation (ES) in seconds.
        fs (float): Sampling frequency in Hz.

    Returns:
        None
    """
    x_axis_trial_s = calculate_scale(data[0], fs)

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(x_axis_trial_s, data[0][1]) #[0][0] is for cuff data and trial 1 
    axs[1].plot(x_axis_trial_s, np.zeros(len(x_axis_trial_s)))
    axs[1].hlines([-0.15, 0.15], xmin=US_start, xmax=US_start+US_length, colors='red', linewidth=3)  # US
    axs[1].text(US_start + US_length / 2, -0.20, 'US', color='red', ha='center', fontsize=10)
    axs[1].vlines(ES, ymin=np.min(data[0][0]), ymax=np.max(data[0][0]), colors='black')  # ES
    axs[0].set_ylim([-0.3,0.3])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylim([-0.3, 0.3])
    axs[1].set_ylabel('Amplitude (V)')
    axs[0].set_title(f'Trial 1: {experiment_param}')
    fig.suptitle(f"Experiment {experiment_number+1} design VS cuff recording", fontsize=15, weight='bold')
    #plt.show()
    plt.savefig(f"figures/experiment_design/exp_{experiment_number+1}.png")
    plt.close()


def plot_ap_superposition(action_potentials, fs, experiment_number, feature):
    """
    Visualize action potentials (APs) for each trial and recording and superimpose them to compare.

    Args:
        action_potentials (list of lists): 2D list of extracted action potentials.
            The first index corresponds to the trial, and the second index corresponds to the ES-provoked AP.
        fs (float): Sampling frequency in Hz.
        experiment_number (int): The experiment number.

    Returns:
        None
    """
    rows = 1
    cols = 3

    x_axis = calculate_scale(action_potentials[0], fs)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharex=True, sharey='row')
    fig.suptitle(f"Experiment {experiment_number+1} inter-trial AP superimposition", fontsize=13, weight='bold')

    for i in range(rows):
        axes[0].plot(x_axis, action_potentials[0][0], color='cyan', label = 'Trial 1, first ES')
        axes[0].plot(x_axis, action_potentials[1][0], color='orange', label = 'Trial 2, first ES')
        axes[0].plot(x_axis, action_potentials[2][0], color='green', label = 'Trial 3, first ES')

        axes[1].plot(x_axis, action_potentials[0][5], color='cyan', label = 'Trial 1, ES = 6')
        axes[1].plot(x_axis, action_potentials[1][5], color='orange', label = 'Trial 2, ES = 6')
        axes[1].plot(x_axis, action_potentials[2][5], color='green', label = 'Trial 3, ES = 6')
    
        axes[2].plot(x_axis, action_potentials[0][1], color='cyan', label = 'Trial 1, ES = 1')
        axes[2].plot(x_axis, action_potentials[0][2], color='orange', label = 'Trial 1, ES = 2')
        axes[2].plot(x_axis, action_potentials[0][3], color='green', label = 'Trial 1, ES = 3')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[1].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (V)')
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    plt.savefig(f"figures/ap_superposition/exp{experiment_number+1}_{feature}.png")
    plt.close()


def plot_single_action_potentials(action_potentials, trial_indices, fs, experiment_number):
    """
    Plot action potentials (APs) for specific trials and ES indices.

    Args:
        action_potentials (list of lists): 2D list of extracted action potentials.
            The first index corresponds to the trial, and the second index corresponds to the ES-provoked AP.
        trial_indices (list of int): Indices of trials to plot.
        fs (float): Sampling frequency in Hz.
        experiment_number (int): The experiment number.

    Returns:
        None
    """
    num_trials = len(action_potentials)
    num_indices = len(trial_indices)
    x_axis= calculate_scale(action_potentials[0], fs)
    
    fig, axes = plt.subplots(1, num_indices, figsize=(20, 6), sharex=True, sharey=True)
    fig.suptitle(f"Experiment {experiment_number}: APs", fontsize=13, weight='bold')

    for i, idx in enumerate(trial_indices):
        for j in range(num_trials):
            axes[i].plot(x_axis, action_potentials[j][idx], label=f'Trial {j+1}')

        axes[i].set_title(f"ES = {idx+1}")
        axes[i].legend()

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.tight_layout()
    plt.show()
    #plt.savefig("figures/exp_aps.png")


def plot_latencies(action_potentials, thresholds, threshold_x_values, latencies, fs, experiment_number, recording):
    """
    Plot each action potential along with the threshold and where it starts.

    Args:
        action_potentials (numpy.ndarray): A 2D array where each row represents an individual action potential.
        thresholds (numpy.ndarray): An array containing the threshold value for each action potential.
        latencies (numpy.ndarray): An array containing the latency of each action potential.
        fs (float): Sampling frequency in Hz.
        experiment_number (int): The experiment number.

    Returns:
        None
    """
    num_ap = len(action_potentials)
    time_axis = calculate_scale(action_potentials, fs)  # Assuming all trials have the same time axis

    # Determine the number of rows and columns for subplots
    rows = num_ap
    cols = 1

    fig, axes = plt.subplots(rows, cols, figsize=(8, 2*num_ap), sharex=True)

    for i in range(num_ap):
        ax = axes[i]
        ap = action_potentials[i]
        ax.plot(time_axis, ap, color='black', label='Action Potential')
        
        # Find the index of the threshold
        threshold_index = np.argmax(ap >= thresholds[i])
        
        # Calculate start and end x points
        end_x = threshold_x_values[i]
        start_x = end_x - latencies[i]

        # Plot everything between start and end x points in yellow
        ax.fill_between(time_axis, min(ap), max(ap), where=(time_axis <= end_x) & (time_axis >= start_x), color='yellow', alpha=0.5, label='Latency')

        # Plot the threshold as a red dot
        ax.plot(threshold_x_values[i], thresholds[i], marker='o', markersize=5, color='red', label='Threshold')
        
        # Plot the y-axis at y=0
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        
        ax.set_title(f'Action Potential {i+1}')
        ax.legend()

    # Set common labels and title
    fig.suptitle(f"Experiment {experiment_number+1} - Action Potentials, Thresholds, and Latencies", fontsize=13, weight='bold')
    axes[-1].set_xlabel('Time (s)')
    axes[num_ap//2].set_ylabel('Amplitude (V)')  # Set ylabel for middle subplot

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"figures/latencies/exp{experiment_number+1}_{recording}.png")
    plt.close()

def plot_length_response(action_potentials, thresholds, threshold_x_values, lengths, fs, experiment_number, recording):
    """
    Plot each action potential along with the threshold and where it starts.

    Args:
        action_potentials (numpy.ndarray): A 2D array where each row represents an individual action potential.
        thresholds (numpy.ndarray): An array containing the threshold value for each action potential.
        latencies (numpy.ndarray): An array containing the latency of each action potential.
        fs (float): Sampling frequency in Hz.
        experiment_number (int): The experiment number.

    Returns:
        None
    """
    num_ap = len(action_potentials)
    time_axis = calculate_scale(action_potentials, fs)  # Assuming all trials have the same time axis

    # Determine the number of rows and columns for subplots
    rows = num_ap
    cols = 1

    fig, axes = plt.subplots(rows, cols, figsize=(8, 2*num_ap), sharex=True)

    for i in range(num_ap):
        ax = axes[i]
        ap = action_potentials[i]
        ax.plot(time_axis, ap, color='black', label='Action Potential')
        
        # Find the index of the threshold
        threshold_index = np.argmax(ap >= thresholds[i])
        
        # Calculate start and end x points
        end_x = threshold_x_values[i] +lengths[i]
        start_x = threshold_x_values[i]

        # Plot everything between start and end x points in yellow
        ax.fill_between(time_axis, min(ap), max(ap), where=(time_axis <= end_x) & (time_axis >= start_x), alpha=0.5, label='length of response')

        # Plot the threshold as a red dot
        ax.plot(threshold_x_values[i], thresholds[i], marker='o', markersize=5, color='red', label='Threshold')
        
        # Plot the y-axis at y=0
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        
        ax.set_title(f'Action Potential {i+1}')
        ax.legend()

    # Set common labels and title
    fig.suptitle(f"Experiment {experiment_number+1} - Action Potentials lengths", fontsize=13, weight='bold')
    axes[-1].set_xlabel('Time (s)')
    axes[num_ap//2].set_ylabel('Amplitude (V)')  # Set ylabel for middle subplot

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"figures/length_response/exp{experiment_number+1}_{recording}.png")
    plt.close()


def plot_fourier_transform(xf, yf, max_freq=700):
    """
    Plots the Fourier Transform of the signal up to a specified maximum frequency.
    
    Args:
        xf (array-like): Frequency bins.
        yf (array-like): Amplitude of the Fourier Transform.
        max_freq (float): Maximum frequency to plot.
    """
    # Find the index where frequency exceeds max_freq
    max_freq_idx = np.where(xf > max_freq)[0][0]
    
    # Limit the xf and yf arrays to max_freq
    xf_limited = xf[:max_freq_idx]
    yf_limited = yf[:max_freq_idx]
    
    # Plot the limited Fourier Transform
    plt.plot(xf_limited, yf_limited)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum of Action Potentials')
    plt.grid()
    plt.show()

def plot_psd(frequencies, psd, experiment, feature, cutoff_freq=500):
    """
    Plot the power spectral density (PSD) and save the plot.
    
    Args:
        frequencies (array-like): The frequencies.
        psd (array-like): The power spectral density values.
        experiment (str): The name of the experiment.
        feature (str): The name of the feature.
        cutoff_freq (float): The cutoff frequency to stop plotting.
    """
    plt.figure()
    plt.semilogy(frequencies, psd)
    plt.title(f'Mean PSD for {experiment} of the {feature} recording')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.grid()
    plt.xlim(left=0,right=cutoff_freq)  # Set the x-axis limit to the cutoff frequency
    plt.savefig(f"figures/power_density_functions/{experiment}_{feature}.png")
    plt.close()


