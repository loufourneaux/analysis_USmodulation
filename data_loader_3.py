import numpy as np
import math
import pandas as pd
import adi

class BiomedicalDataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.channels_name = []
        self.recording_id_list = []
        self.exp_boundary = []
        self.n_samples_list = []
        self.dt_list = []
        self.comments_list = []
        self.data_cuff = []
        self.data_ta = []
        self.data_gn = []
        self.fs = None
        self.df = None

    def load_data(self):
        # Open .adicht file
        f = adi.read_file(self.filename)

        # Declare and sort data
        for channel in f.channels:
            self.channels_name.append(channel.name)

        ch_cuff = self.channels_name.index('Cuff')
        ch_ta = self.channels_name.index('TA')
        ch_gn = self.channels_name.index('GM')

        for idx, recording in enumerate(f.channels[ch_cuff].records):
            self.recording_id_list.append(recording.id)
            self.n_samples_list.append(recording.n_ticks)
            self.dt_list.append(recording.tick_dt)

            if len(recording.comments) > 0:
                self.comments_list.append(recording.comments[0].text)
                self.exp_boundary.append(recording.id)

        # Calculate sampling frequency
        if len(set(self.dt_list)) == 1:
            dt = self.dt_list[0]
            self.fs = math.ceil(1/dt)

        # Form unique dataset
        for recording_id in self.recording_id_list:
            self.data_cuff.append(f.channels[ch_cuff].get_data(recording_id))
            self.data_ta.append(f.channels[ch_ta].get_data(recording_id))
            self.data_gn.append(f.channels[ch_gn].get_data(recording_id))

        # Organize data into DataFrame
        num_experiments = len(self.exp_boundary)
        data_organization = {
            "experiment": [i for i in range(num_experiments)],
            "exp boundary": [[self.exp_boundary[i-1], self.exp_boundary[i]] for i in range(num_experiments)],
            "label": [self.comments_list[i] for i in range(num_experiments)]
        }
        self.df = pd.DataFrame(data_organization)

    def process_all_experiments(self, start_exp, end_exp):
        results = {}
        for exp_index in range(start_exp, end_exp+1):
            exp_name = f"exp{exp_index:02d}"
            results[exp_name] = self.concatenate_experiment_data(self.data_cuff, self.data_ta, self.data_gn, self.exp_boundary, exp_index-1)
        return results
    
    def concatenate_experiment_data(self, data_cuff, data_ta, data_gn, exp_boundaries, exp_number):
        """
        Concatenates the data for a specific experiment.

        Parameters:
        data_cuff (list): List of data arrays for the cuff.
        data_ta (list): List of data arrays for the TA.
        data_gn (list): List of data arrays for the GN.
        exp_boundaries (list): List of indices indicating the boundaries of each experiment.
        exp_number (int): The experiment number to concatenate data for.

        Returns:
        tuple: Three arrays containing concatenated data for cuff, TA, and GN for the specified experiment.
        """
        start, end = exp_boundaries[exp_number - 1], exp_boundaries[exp_number]
        exp_cuff = np.concatenate(data_cuff[start:end])
        exp_ta = np.concatenate(data_ta[start:end])
        exp_gn = np.concatenate(data_gn[start:end])
        return exp_cuff, exp_ta, exp_gn
        
    def get_number_trials(self, experiment):
        """
        Get the number of trials for a specific experiment.

        Parameters:
            experiment (str): The name of the experiment.

        Returns:
            int: The number of trials for the specified experiment.
        """
        # Ensure experiment name format is valid
        if not experiment.startswith("exp"):
            print("Invalid experiment name format.")
            return None
    
        exp_index = int(experiment[3:])
    
        # Ensure exp_index is within range
        if exp_index < 1 or exp_index > len(self.exp_boundary):
            print("Experiment index out of range.")
            return None

        start_idx = self.exp_boundary[exp_index - 1]
        end_idx = self.exp_boundary[exp_index] if exp_index < len(self.exp_boundary) else len(self.data_cuff)
        num_trials = end_idx - start_idx

        return num_trials

        

