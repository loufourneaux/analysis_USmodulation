from data_loader_3 import *
from preprocessing import *
from visualization import *
from processing import *
from statistic_tests import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

##LOAD DATA
filename = 'data_20240130 inhibit.adicht'
loader = BiomedicalDataLoader(filename)
loader.load_data()

#create dictionnary to store information on all the experiments 
#Initialize dictionaries to store modulation values and experiment parameters
parameters_numbers=np.array([])
#these are the features we are extracting for each experiment which we study outside the loop. Can add features to the dictionaries
features_cuff = {'latency_ES': [], 'latency_US': [],'amplitude_ES': [], 'amplitude_US': [],'area_ES': [], 'area_US': [],'length_ES':[], 'length_US':[], 'psd':[]}
features_ta = {'latency_ES': [], 'latency_US': [],'amplitude_ES': [], 'amplitude_US': [],'area_ES': [], 'area_US': [],'length_ES':[], 'length_US':[], 'psd':[]}
features_gm = {'latency_ES': [], 'latency_US': [],'amplitude_ES': [], 'amplitude_US': [],'area_ES': [], 'area_US': [],'length_ES':[], 'length_US':[],  'psd':[]}


#comment and uncomment the plot functions to save the plots 

for experiment_number in [14,15]:  # Update with the experiment numbers you want to compare
    #chose experiment to study
    experiment= f"exp{experiment_number+1}" 
    experiment_parameters = loader.df["label"][experiment_number]
    #store the parameters numbers for later stats
    parameters_numbers = np.append(parameters_numbers,get_number_parameters(experiment_parameters))
    n_parts= loader.get_number_trials(experiment)

    #sometimes may have to determine the n_parts by hand as they are not always the right one, can do as below
    #if experiment_number==25:
        #n_parts=2
    #else:
       #n_parts=3
        
    all_experiments = loader.process_all_experiments(14, len(loader.exp_boundary)) # because from 14 it is the right protocol design 
    exp_data= all_experiments[experiment]

    # Loop through exp_data and apply filter_data to each element
    filtered_exp_data = [filter_data(channel_data, loader.fs) for channel_data in exp_data]
    segmented_exp_data = [segment_data(channel_data, n_parts) for channel_data in filtered_exp_data]

    #see experiment overview 
    plot_experiment_overview(segmented_exp_data, experiment_number, experiment_parameters, n_parts, loader.fs)

    #design the experiment depending on the values 
    ES, US_start, US_length= chose_protocol(1) #1 for 2 bursts, 2 for 3 bursts
    ##PLOT DESIGN EXPERIMENT
    plot_design_experiment(segmented_exp_data,experiment_number, experiment_parameters,US_start,US_length, ES, loader.fs)

    ##FEATURE EXTRACTION FOR EACH RECORDING
    #cuff ap for each trial
    aps_cuff=[] 
    lat_cuff=[] #of shape (3, 3, 15)
    area_cuff =[] #of shape (3, 15)
    ampli_cuff=[] # of shape (3,15)
    length_cuff=[] #of shape (3,15)
    norm_aps_cuff=[]
    psd_cuff=[] #of shape (3,-)
    length_resp_cuff=[]#of shape (3,15)
    for i in range(n_parts):
        ap_cuff= extract_action_potentials(segmented_exp_data[0][i], ES, loader.fs, w=0.1)
        aps_cuff.append(ap_cuff)
        norm_ap_cuff=normalize_action_potentials(ap_cuff)
        norm_aps_cuff.append(norm_ap_cuff)
        threshold, threshold_x_value, latency=find_latencies(norm_ap_cuff, calculate_scale(norm_ap_cuff, loader.fs))
        end_x_value=find_last_crossing_time(norm_ap_cuff, calculate_scale(norm_ap_cuff, loader.fs))
        length_resp_cuff.append(end_x_value-np.array(threshold_x_value))
        lat_cuff.append([threshold, threshold_x_value, latency])
        area_cuff.append(area_under_curve(norm_ap_cuff, loader.fs))
        ampli_cuff.append(calculate_peak_to_peak_distances(ap_cuff))
        frequencies, psd = welch(segmented_exp_data[0][i], fs=loader.fs, nperseg=1024)
        psd_cuff.append(psd)
  
    
    #average the features of ES and US peaks for each trials 
    lat_cuff=np.array(lat_cuff)
    area_cuff = np.array(area_cuff)
    ampli_cuff = np.array(ampli_cuff)
    length_resp_cuff=np.array(length_resp_cuff)

    # Average the features of ES and US peaks across the trials and add them to the dictionaries
    features_cuff['latency_ES'].append(np.mean(lat_cuff[:, 2, :5], axis=0))
    features_cuff['latency_US'].append(np.mean(lat_cuff[:, 2, 5:15], axis=0))
    features_cuff['area_ES'].append(np.mean(area_cuff[:, :5], axis=0))
    features_cuff['area_US'].append(np.mean(area_cuff[:, 5:15], axis=0))
    features_cuff['amplitude_ES'].append(np.mean(ampli_cuff[:, :5], axis=0))
    features_cuff['amplitude_US'].append(np.mean(ampli_cuff[:, 5:15], axis=0))
    features_cuff['psd'].append(np.mean(psd_cuff, axis=0))
    features_cuff['length_ES'].append(np.mean(length_resp_cuff[:,:5], axis=0))
    features_cuff['length_US'].append(np.mean(length_resp_cuff[:, 5:15], axis=0))

    #save plot
    #plot_ap_superposition(aps_cuff, loader.fs, experiment_number, 'cuff')
    #plot_psd(frequencies, np.mean(psd_cuff, axis=0), experiment, 'cuff')
    #plot_latencies(norm_aps_cuff[1], lat_cuff[1,0,:], lat_cuff[1,1,:],lat_cuff[1,2,:], loader.fs, experiment_number, 'cuff')#latemcies for the first trial (modify the 0 to get the other trials)
    #plot_length_response(norm_aps_cuff[1], lat_cuff[1,0,:], lat_cuff[1,1,:],length_resp_cuff[1], loader.fs, experiment_number, 'cuff')

    # ta ap for each trial
    aps_ta = []
    lat_ta = []  # of shape (3, 3, 15)
    area_ta = []  # of shape (3, 15)
    ampli_ta = []  # of shape (3,15)
    length_ta = []  # of shape (3,15)
    norm_aps_ta = []
    psd_ta=[] #of shape (3,-)
    length_resp_ta=[]#of shape (3,15)
    for i in range(n_parts):
        ap_ta = extract_action_potentials(segmented_exp_data[1][i], ES, loader.fs, w=0.1)
        aps_ta.append(ap_ta)
        norm_ap_ta = normalize_action_potentials(ap_ta)
        norm_aps_ta.append(norm_ap_ta)
        threshold, threshold_x_value, latency = find_latencies(norm_ap_ta, calculate_scale(norm_ap_ta, loader.fs))
        end_x_value=find_last_crossing_time(norm_ap_ta, calculate_scale(norm_ap_ta, loader.fs))
        length_resp_ta.append(end_x_value-np.array(threshold_x_value))
        lat_ta.append([threshold, threshold_x_value, latency])
        area_ta.append(area_under_curve(norm_ap_ta, loader.fs))
        ampli_ta.append(calculate_peak_to_peak_distances(ap_ta))
        frequencies, psd = welch(segmented_exp_data[1][i], fs=loader.fs, nperseg=1024)
        psd_ta.append(psd)

    # average the features of ES and US peaks for each trials
    lat_ta = np.array(lat_ta)
    area_ta = np.array(area_ta)
    ampli_ta = np.array(ampli_ta)
    length_resp_ta=np.array(length_resp_ta)

    # Average the features of ES and US peaks for each trial and add them to the dictionaries
    features_ta['latency_ES'].append(np.mean(lat_ta[:, 2, :5], axis=0))
    features_ta['latency_US'].append(np.mean(lat_ta[:, 2, 5:15], axis=0))
    features_ta['area_ES'].append(np.mean(area_ta[:, :5], axis=0))
    features_ta['area_US'].append(np.mean(area_ta[:, 5:15], axis=0))
    features_ta['amplitude_ES'].append(np.mean(ampli_ta[:, :5], axis=0))
    features_ta['amplitude_US'].append(np.mean(ampli_ta[:, 5:15], axis=0))
    features_ta['psd'].append(np.mean(psd_ta, axis=0))
    features_ta['length_ES'].append(np.mean(length_resp_ta[:,:5], axis=0))
    features_ta['length_US'].append(np.mean(length_resp_ta[:, 5:15], axis=0))


    # save plot
    #plot_ap_superposition(aps_ta, loader.fs, experiment_number, 'TA')
    #plot_psd(frequencies, np.mean(psd_ta, axis=0), experiment, 'TA')
    #plot_latencies(norm_aps_ta[1], lat_ta[1,0,:], lat_ta[1,1,:],lat_ta[1,2,:], loader.fs, experiment_number, 'TA')#latemcies for the first trial (modify the 0 to get the otehr trials)
    #plot_length_response(norm_aps_ta[1], lat_ta[1,0,:], lat_ta[1,1,:],length_resp_ta[1], loader.fs, experiment_number, 'TA')

    # gm ap for each trial
    aps_gm = []
    lat_gm = []  # of shape (3, 3, 15)
    area_gm = []  # of shape (3, 15)
    ampli_gm = []  # of shape (3,15)
    length_gm = []  # of shape (3,15)
    norm_aps_gm = []
    psd_gm =[]
    length_resp_gm=[]#of shape (3,15)
    for i in range(n_parts):
        ap_gm = extract_action_potentials(segmented_exp_data[2][i], ES, loader.fs, w=0.1)
        aps_gm.append(ap_gm)
        norm_ap_gm = normalize_action_potentials(ap_gm)
        norm_aps_gm.append(norm_ap_gm)
        threshold, threshold_x_value, latency = find_latencies(norm_ap_gm, calculate_scale(norm_ap_gm, loader.fs))
        end_x_value=find_last_crossing_time(norm_ap_gm, calculate_scale(norm_ap_gm, loader.fs))
        length_resp_gm.append(end_x_value-np.array(threshold_x_value))
        lat_gm.append([threshold, threshold_x_value, latency])
        area_gm.append(area_under_curve(norm_ap_gm, loader.fs))
        ampli_gm.append(calculate_peak_to_peak_distances(ap_gm))
        frequencies, psd = welch(segmented_exp_data[2][i], fs=loader.fs, nperseg=1024)
        psd_gm.append(psd)

    # average the features of ES and US peaks for each trials
    lat_gm = np.array(lat_gm)
    area_gm = np.array(area_gm)
    ampli_gm = np.array(ampli_gm)
    length_resp_gm=np.array(length_resp_gm)

    # Average the features of ES and US peaks for each trial and add them to the dictionaries
    features_gm['latency_ES'].append(np.mean(lat_gm[:, 2, :5], axis=0))
    features_gm['latency_US'].append(np.mean(lat_gm[:, 2, 5:15], axis=0))
    features_gm['area_ES'].append(np.mean(area_gm[:, :5], axis=0))
    features_gm['area_US'].append(np.mean(area_gm[:, 5:15], axis=0))
    features_gm['amplitude_ES'].append(np.mean(ampli_gm[:, :5], axis=0))
    features_gm['amplitude_US'].append(np.mean(ampli_gm[:, 5:15], axis=0))
    features_gm['psd'].append(np.mean(psd_gm, axis=0))
    features_gm['length_ES'].append(np.mean(length_resp_gm[:,:5], axis=0))
    features_gm['length_US'].append(np.mean(length_resp_gm[:, 5:15], axis=0))


    # save plot
    #plot_ap_superposition(aps_gm, loader.fs, experiment_number, 'GM')
    #plot_psd(frequencies, np.mean(psd_gm, axis=0), experiment, 'GM')
    #plot_latencies(norm_aps_gm[0], lat_gm[0,0,:], lat_gm[0,1,:],lat_gm[0,2,:], loader.fs, experiment_number, 'GM')#latemcies for the first trial (modify the 0 to get the other trials)
    #plot_length_response(norm_aps_gm[1], lat_gm[1,0,:], lat_gm[1,1,:],length_resp_gm[1], loader.fs, experiment_number, 'GM')

    #determine the modulation across sonication length 
    features = ['latency', 'amplitude', 'area']
    #plot_sonication_effect(features_cuff, features_ta, features_gm, features, ES[5:], US_start, US_length, experiment_number) #will plot only the first one so run the loop for only the one you want to plot
   

##SEE DIFFERENCES OF MODULATION DEPENDING ON EXPERIMENT AKA PARAMETERS
#remove outliers 
# Remove outliers from feature dictionaries
features_cuff = remove_outliers_from_dict(features_cuff)
features_ta = remove_outliers_from_dict(features_ta)
features_gm = remove_outliers_from_dict(features_gm)

#test difference between ES and US features 
significant_tests = run_all_significance_tests(features_ta)
print("Significant test results:", significant_tests)
significant_tests = run_all_significance_tests(features_cuff)
print("Significant test results:", significant_tests)
significant_tests = run_all_significance_tests(features_gm)
print("Significant test results:", significant_tests)

#plot each feature value across time 
#plot_across_time(features_cuff, features_ta, features_gm, 'length', ES, US_start, US_length) #change 'length' to get another feature like 'latency', and adjust some ylim parameters in the function for a better plot

#mean modulation of each experiment and recording plot 
features = ['latency', 'amplitude', 'area', 'length'] #add features here if added some in the dictionnaries 
plot_modulation(features_cuff, features_ta, features_gm, features)#as boxplot
plot_three_var(features_cuff, features_ta, features_gm, features)#as scatterplot

#ANOVA to test for difference between experiments
results = perform_all_anova(features_cuff, features_ta, features_gm, features)
# Save results to a text file
#rename the other files so it doesn't overwrite it
with open('ANOVA/anova_results.txt', 'w') as file:
    for feature, result in results.items():
        file.write(f"Feature: {feature}\n")
        for recording, stats in result.items():
            file.write(f"{recording} ANOVA: F-statistic = {stats['ANOVA']['F-statistic']}, p-value = {stats['ANOVA']['p-value']}\n")
            if 'Tukey' in stats:
                file.write(f"{recording} Tukey's HSD test:\n{stats['Tukey']}\n")
        file.write("\n")

#test statistical difference of the power spectral density functions between experiments, for each recording 
# Save results to a text file
results = perform_all_psd_anova(features_cuff, features_ta, features_gm)
with open('ANOVA/psd_anova_results.txt', 'w') as file:
    for recording, stats in results.items():
        file.write(f"{recording} ANOVA: F-statistic = {stats['ANOVA']['F-statistic']}, p-value = {stats['ANOVA']['p-value']}\n")
        if 'Tukey' in stats:
            file.write(f"{recording} Tukey's HSD test:\n{stats['Tukey']}\n")
        file.write("\n")
