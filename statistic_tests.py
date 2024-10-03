import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
from processing import *


def modulation(ES_values, US_values):
    """
    Determines the normalized difference between each value under US sonication and the mean value under only electrical stimulation
    """
    ES_mean= np.mean(ES_values)
    return ((US_values-ES_mean)/ES_mean)

def mean_std_modulation(ES_values, US_values):
    """
    Calculate the mean and standard deviation of the normalized modulation of electrical stimulation by ultrasound 

    Parameters:
        ES_values (np.array): shape (n_parts,)
        US_values (np.array): shape(n_parts,)
    
    Returns:
        np.array of same shape
    """
    mean=np.mean((US_values-ES_values)/ES_values)
    std = np.std((US_values-ES_values)/ES_values)

    return mean, std

def calculate_modulations(feature_cuff, feature_ta, feature_gm, feature ):
    """
    Calculate the modulation of the feature by ultrasound for each recording

    Returns:
        Three lists of nb_experiments np.arrays containing the modulation for each action potential
    """
    nb_experiments=len(feature_gm[f'{feature}_ES'])

    mod_cuff= []
    mod_ta=[]
    mod_gm=[]
    for i in range(nb_experiments):
        mod_cuff.append(modulation(feature_cuff[f'{feature}_ES'][i],feature_cuff[f'{feature}_US'][i]))
        mod_ta.append(modulation(feature_ta[f'{feature}_ES'][i],feature_ta[f'{feature}_US'][i]))
        mod_gm.append(modulation(feature_gm[f'{feature}_ES'][i],feature_gm[f'{feature}_US'][i]))
    
    return mod_cuff, mod_ta, mod_gm

def perform_anova(*modulations):
    """
    Perform ANOVA on the provided lists of modulation arrays.

    Parameters:
        *modulations: Variable number of lists containing modulation arrays.

    Returns:
        F-statistic and p-value for the ANOVA test.
    """
    f_statistic, p_value = f_oneway(*modulations)
    return f_statistic, p_value

def perform_tukey_test(modulations):
    """
    Perform Tukey's HSD test on the provided modulation arrays.

    Parameters:
        modulations: List of modulation arrays.

    Returns:
        Tukey's HSD test results.
    """
    combined_data = np.concatenate(modulations)
    labels = np.array([group_id for group_id, mod in enumerate(modulations) for _ in mod])
    
    tukey_result = pairwise_tukeyhsd(endog=combined_data, groups=labels, alpha=0.05)
    return tukey_result

def perform_all_anova(features_cuff, features_ta, features_gm, features):
    """
    Perform ANOVA for all features and recordings, followed by Tukey's HSD test if ANOVA is significant.

    Parameters:
        features_cuff, features_ta, features_gm: Dictionaries containing data for each feature and recording.
        features: List of feature names to analyze.

    Returns:
        Dictionary containing ANOVA and Tukey's HSD test results for each feature and each recording type.
    """
    results = {}
    for feature in features:
        mod_cuff, mod_ta, mod_gm = calculate_modulations(features_cuff, features_ta, features_gm, feature)
        
        f_statistic_cuff, p_value_cuff = perform_anova(*mod_cuff)
        f_statistic_ta, p_value_ta = perform_anova(*mod_ta)
        f_statistic_gm, p_value_gm = perform_anova(*mod_gm)

        results[feature] = {
            'Cuff': {'ANOVA': {'F-statistic': f_statistic_cuff, 'p-value': p_value_cuff}},
            'TA': {'ANOVA': {'F-statistic': f_statistic_ta, 'p-value': p_value_ta}},
            'GM': {'ANOVA': {'F-statistic': f_statistic_gm, 'p-value': p_value_gm}}
        }

        if p_value_cuff < 0.05:
            tukey_result_cuff = perform_tukey_test(mod_cuff)
            results[feature]['Cuff']['Tukey'] = tukey_result_cuff.summary()
        
        if p_value_ta < 0.05:
            tukey_result_ta = perform_tukey_test(mod_ta)
            results[feature]['TA']['Tukey'] = tukey_result_ta.summary()

        if p_value_gm < 0.05:
            tukey_result_gm = perform_tukey_test(mod_gm)
            results[feature]['GM']['Tukey'] = tukey_result_gm.summary()

    return results

def perform_all_psd_anova(features_cuff, features_ta, features_gm):
    """
    Perform ANOVA for all PSDs and recordings, followed by Tukey's HSD test if ANOVA is significant.

    Parameters:
        features_cuff, features_ta, features_gm: Dictionaries containing PSD data.

    Returns:
        Dictionary containing ANOVA and Tukey's HSD test results for PSDs in each recording type.
    """
    results = {}

    # Perform ANOVA for Cuff
    f_statistic_cuff, p_value_cuff = perform_anova(*features_cuff['psd'])
    results['Cuff'] = {'ANOVA': {'F-statistic': f_statistic_cuff, 'p-value': p_value_cuff}}
    if p_value_cuff < 0.05:
        tukey_result_cuff = perform_tukey_test(features_cuff['psd'])
        results['Cuff']['Tukey'] = tukey_result_cuff.summary()
    
    # Perform ANOVA for TA
    f_statistic_ta, p_value_ta = perform_anova(*features_ta['psd'])
    results['TA'] = {'ANOVA': {'F-statistic': f_statistic_ta, 'p-value': p_value_ta}}
    if p_value_ta < 0.05:
        tukey_result_ta = perform_tukey_test(features_ta['psd'])
        results['TA']['Tukey'] = tukey_result_ta.summary()
    
    # Perform ANOVA for GM
    f_statistic_gm, p_value_gm = perform_anova(*features_gm['psd'])
    results['GM'] = {'ANOVA': {'F-statistic': f_statistic_gm, 'p-value': p_value_gm}}
    if p_value_gm < 0.05:
        tukey_result_gm = perform_tukey_test(features_gm['psd'])
        results['GM']['Tukey'] = tukey_result_gm.summary()

    return results



def plot_modulation(features_cuff, features_ta, features_gm, features):
    n_experiments = len(features_cuff[f'{features[0]}_ES'])
    experiments = np.arange(n_experiments)
    fig, axs = plt.subplots(1, len(features), figsize=(15, 5))

    # Pastel colors
    colors = {'Cuff': '#548FD6', 'TA': '#75AF61', 'GM': '#F57F38'}  # Pastel blue, pastel orange, pastel green
    
    for i, feature in enumerate(features):
        mod_cuff, mod_ta, mod_gm = calculate_modulations(features_cuff, features_ta, features_gm, feature)

        means_cuff = [np.mean(mod) for mod in mod_cuff]
        std_cuff = [np.std(mod) for mod in mod_cuff]

        means_ta = [np.mean(mod) for mod in mod_ta]
        std_ta = [np.std(mod) for mod in mod_ta]

        means_gm = [np.mean(mod) for mod in mod_gm]
        std_gm = [np.std(mod) for mod in mod_gm]

        ax = axs[i] if len(features) > 1 else axs
        ax.bar(experiments - 0.25, means_cuff, 0.25, yerr=std_cuff, capsize=5, alpha=0.7, label='Cuff', color=colors['Cuff'])
        ax.bar(experiments, means_ta, 0.25, yerr=std_ta, capsize=5, alpha=0.7, label='TA', color=colors['TA'])
        ax.bar(experiments + 0.25, means_gm, 0.25, yerr=std_gm, capsize=5, alpha=0.7, label='GM', color=colors['GM'])

        ax.set_xlabel('Experiments')
        ax.set_ylabel(f'Mean normalized {feature.capitalize()} Modulation')
        ax.axhline(0, color='gray', linestyle='--')  # Add dotted line at y = 0
        ax.set_xticks(experiments)
        ax.legend()
        

    fig.suptitle('Modulation of amplitude, AUC, length and latency of response under US sonication', fontsize=14, weight='bold')            
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'figures/US_modulation/barplot.png') #change title with parameters before plotting or after
    plt.close()


def plot_modulation_significant(features_cuff, features_ta, features_gm, features):
    # Run significance tests
    significant_tests_cuff = run_all_significance_tests(features_cuff)
    significant_tests_ta = run_all_significance_tests(features_ta)
    significant_tests_gm = run_all_significance_tests(features_gm)

    n_experiments = len(features_cuff[f'{features[0]}_ES'])
    experiments = np.arange(n_experiments)
    fig, axs = plt.subplots(1, len(features), figsize=(15, 5))

    # Pastel colors
    colors = {'Cuff': '#548FD6', 'TA': '#75AF61', 'GM': '#F57F38'}  # Pastel blue, pastel orange, pastel green

    for i, feature in enumerate(features):
        mod_cuff, mod_ta, mod_gm = calculate_modulations(features_cuff, features_ta, features_gm, feature)

        means_cuff = [np.mean(mod) for mod in mod_cuff]
        std_cuff = [np.std(mod) for mod in mod_cuff]

        means_ta = [np.mean(mod) for mod in mod_ta]
        std_ta = [np.std(mod) for mod in mod_ta]

        means_gm = [np.mean(mod) for mod in mod_gm]
        std_gm = [np.std(mod) for mod in mod_gm]

        ax = axs[i] if len(features) > 1 else axs

        # Plot significant differences only
        significant_experiments_cuff = significant_tests_cuff.get((f'{feature}_ES', f'{feature}_US'), [])
        significant_experiments_ta = significant_tests_ta.get((f'{feature}_ES', f'{feature}_US'), [])
        significant_experiments_gm = significant_tests_gm.get((f'{feature}_ES', f'{feature}_US'), [])

        if significant_experiments_cuff:
            ax.bar([e - 0.25 for e in significant_experiments_cuff], 
                   [means_cuff[e] for e in significant_experiments_cuff], 
                   0.25, yerr=[std_cuff[e] for e in significant_experiments_cuff], 
                   capsize=5, alpha=0.7, label='Cuff', color=colors['Cuff'])
        
        if significant_experiments_ta: 
            ax.bar(significant_experiments_ta, [means_ta[e] for e in significant_experiments_ta], 0.25, yerr=[std_ta[e] for e in significant_experiments_ta], capsize=5, alpha=0.7, label='TA', color=colors['TA'])
        
        if significant_experiments_gm:
            ax.bar([e + 0.25 for e in significant_experiments_gm], 
                   [means_gm[e] for e in significant_experiments_gm], 
                   0.25, yerr=[std_gm[e] for e in significant_experiments_gm], 
                   capsize=5, alpha=0.7, label='GM', color=colors['GM'])

        ax.set_xlabel('Experiments')
        ax.axhline(0, color='gray', linestyle='--')  # Add dotted line at y = 0
        ax.set_ylabel(f'Mean normalized {feature.capitalize()} Modulation')
        ax.set_xticks(experiments)
        ax.legend()

    fig.suptitle('Modulation of amplitude, AUC, length and latency of response under US sonication', fontsize=14, weight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'figures/US_modulation/barplot_significant_only.png')  # Change title with parameters
    plt.close()



def plot_three_var(features_cuff, features_ta, features_gm, features):
    """
    Plots the modulation as a scatterplot
    """
    n_experiments = len(features_cuff[f'{features[0]}_ES'])
    experiments = np.arange(n_experiments)
    fig, axs = plt.subplots(1, len(features), figsize=(15, 5))

    # Pastel colors
    colors = {'Cuff': '#548FD6', 'TA': '#75AF61', 'GM': '#F57F38'}  # Pastel blue, pastel orange, pastel green

    for i, feature in enumerate(features):
        mod_cuff, mod_ta, mod_gm = calculate_modulations(features_cuff, features_ta, features_gm, feature)

        
        means_cuff = [np.mean(mod) for mod in mod_cuff]
        std_cuff = [np.std(mod) for mod in mod_cuff]

        means_ta = [np.mean(mod) for mod in mod_ta]
        std_ta = [np.std(mod) for mod in mod_ta]

        means_gm = [np.mean(mod) for mod in mod_gm]
        std_gm = [np.std(mod) for mod in mod_gm]
        
        ax = axs[i] if len(features) > 1 else axs

        ax.errorbar(experiments, means_cuff, yerr=std_cuff, label='Cuff', fmt='-o', color=colors['Cuff'])
        #ax.errorbar(experiments, means_ta, yerr=std_ta, label='TA', fmt='-s', color=colors['TA'])
        ax.errorbar(experiments, means_gm, yerr=std_gm, label='GM', fmt='-^', color=colors['GM'])


        ax.set_xlabel('Experiments')
        ax.axhline(0, color='gray', linestyle='--')  # Add dotted line at y = 0
        ax.set_ylabel(f'Mean normalized {feature.capitalize()} Modulation')
        ax.set_xticks(experiments)
        ax.legend()
        # Adjusting y-axis limits to ensure visibility of data
        ax.set_ylim(-0.8, 0.8)

    fig.suptitle('Modulation of amplitude, AUC, length and latency of response under US sonication', fontsize=14, weight='bold') 
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'figures/US_modulation/scatterplot.png')
    plt.close()



def stats_difference_significance(group1, group2):

    # Perform the Mann-Whitney U test, can be changed by ttest_ind for ttest
    u_stat, p_value = mannwhitneyu(group1, group2)

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        return True
    else: 
        return False


def run_all_significance_tests(features_ta):
    """
    Run statistical significance tests on specified feature pairs in features_ta dictionary.
    
    Args:
        features_ta (dict): Dictionary containing feature data.
        
    Returns:
        dict: Dictionary containing significant test results.
    """
    tests = [
        ('amplitude_ES', 'amplitude_US'),
        ('area_ES', 'area_US'),
        ('latency_ES', 'latency_US'),
        ('length_ES', 'length_US')
    ]
    
    significant_tests = {}
    
    for feature1, feature2 in tests:
        significant_rows = []
        for i in range(len(features_ta[feature1])):
            group1 = features_ta[feature1][i]
            group2 = features_ta[feature2][i]
            if stats_difference_significance(group1, group2):
                significant_rows.append(i)
        significant_tests[(feature1, feature2)] = significant_rows
    
    return significant_tests



def plot_sonication_effect(features_cuff, features_ta, features_gm, features, ES, US_start, US_length, exp_number):
    fig, axs = plt.subplots(1, len(features), figsize=(15, 5))

    for i, feature in enumerate(features):
        mod_cuff, mod_ta, mod_gm = calculate_modulations(features_cuff, features_ta, features_gm, feature)


        ax = axs[i] if len(features) > 1 else axs

        # Plot the modulation over time for each feature
        colors = {'Cuff': '#548FD6', 'TA': '#75AF61', 'GM': '#F57F38'}  # Pastel blue, pastel orange, pastel green

        ax.plot(ES, mod_cuff[0], 'o', label='Cuff', color=colors['Cuff'])
        ax.plot(ES, mod_ta[0], 's', label='TA', color=colors['TA'])
        ax.plot(ES, mod_gm[0], '^', label='GM', color=colors['GM'])

        
        # Fit regression lines and plot them
        for mod, color in zip([mod_cuff[0], mod_ta[0], mod_gm[0]], [colors['Cuff'], colors['TA'], colors['GM']]):
            coeffs = np.polyfit(ES, mod, 1)  # Fit a linear regression (degree 1 polynomial)
            regression_line = np.polyval(coeffs, ES)
            ax.plot(ES, regression_line, linestyle='-', color=color, linewidth=1)

        # Plot arrow for US duration
        arrow_start = US_start # Starting point of the arrow
        arrow_end = US_start + US_length  # Ending point of the arrow
        arrow_height = -0.3 # Height of the arrow (adjust as needed)
        ax.arrow(arrow_start, arrow_height, arrow_end - arrow_start, 0, head_width=0.1, head_length=0.1, fc='r', ec='r', linestyle='-', label='US Duration')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Normalized {feature.capitalize()} Modulation')
        ax.legend()
        # Adjusting y-axis limits to ensure visibility of data
        ax.set_ylim(-0.4, 1.2)

    fig.suptitle(f'Modulations for Experiment {exp_number+1}', fontsize=13, weight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'figures/US_modulation/along_sonication_{exp_number+1}')
    plt.close()



def plot_across_time(features_cuff, features_ta, features_gm, feature, ES, US_start, US_length):
    n_experiments = len(features_cuff[f'{feature}_ES'])
    fig, axs = plt.subplots(1, n_experiments, figsize=(15, 5))
    
    # Choose metrics
    if feature == 'latency' or feature == 'length':
        metrics = '(s)'
    elif feature == 'amplitude':
        metrics = '(V)'
    else:
        metrics = ''

    colors = {'Cuff': '#548FD6', 'TA': '#75AF61', 'GM': '#F57F38'}  # Pastel blue, pastel green, pastel orange

    for i in range(n_experiments):
        ax = axs[i] 

        # Combine ES and US values and remove outliers
        combined_cuff = np.concatenate((features_cuff[f'{feature}_ES'][i], features_cuff[f'{feature}_US'][i]))
        combined_ta = np.concatenate((features_ta[f'{feature}_ES'][i], features_ta[f'{feature}_US'][i]))
        combined_gm = np.concatenate((features_gm[f'{feature}_ES'][i], features_gm[f'{feature}_US'][i]))

        cleaned_cuff = remove_outliers(combined_cuff)
        cleaned_ta = remove_outliers(combined_ta)
        cleaned_gm = remove_outliers(combined_gm)

        # Plot the modulation over time for each feature
        ax.plot(ES, combined_cuff, 'o', label='Cuff', color=colors['Cuff'])
        ax.plot(ES, combined_ta, 's', label='TA', color=colors['TA'])
        ax.plot(ES, combined_gm, '^', label='GM', color=colors['GM'])

        # Fit regression lines and plot them
        for mod, color, es_values in zip([cleaned_cuff, cleaned_ta, cleaned_gm], 
                                         [colors['Cuff'], colors['TA'], colors['GM']], 
                                         [ES, ES, ES]):
            coeffs = np.polyfit(es_values[:len(mod)], mod, 1)  # Fit a linear regression (degree 1 polynomial)
            regression_line = np.polyval(coeffs, es_values[:len(mod)])
            ax.plot(es_values[:len(mod)], regression_line, linestyle='-', color=color, linewidth=1)

        # Plot arrow for US duration
        arrow_start = US_start  # Starting point of the arrow
        arrow_end = US_start + US_length  # Ending point of the arrow
        arrow_height = -0.01  # Height of the arrow (adjust as needed)
        ax.arrow(arrow_start, arrow_height, arrow_end - arrow_start, 0, head_width=0.01, head_length=0.1, fc='r', ec='r', linestyle='-', label='US Duration')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{feature.capitalize()} {metrics}')
        ax.legend()
        # Adjusting y-axis limits to ensure visibility of data
        ax.set_ylim(-0.04, 0.04)

    fig.suptitle(f'AP {feature} value across time for each experiment', fontsize=13, weight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'figures/US_modulation/{feature}_across_time.png')
    plt.close()