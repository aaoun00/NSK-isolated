import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spikes import SpikeCluster, SpikeTrain
from library.ensemble_space import Cell

def _check_inputs(data, spike_times, interval):
    '''
    This is an amalgamated test function for the compute_firing_rate function and the compute_average_speed function.

    It is best practice to break test functions apart, but for a small project,
    it clearer to have them all in one function.
    '''
    data_dimensionality = data.shape.__len__()
    data_type = type(data)
    timestamp_dimensionality = spike_times[0].shape.__len__()
    timestamp_type = type(spike_times[0])

    if data_dimensionality != 1 or data_type != np.ndarray:
        raise ValueError('data and spike_times must be a 1D numpy array')
        sys.exit("Error: data and spike_times must be a 1D array")
    if timestamp_dimensionality != 1 or timestamp_type != np.ndarray:
        raise ValueError('data and spike_times must be a 1D numpy array contained within an array with a single index of 0')
        sys.exit("Error: data and spike_times must be a 1D numpy array contained within an array with a single index of 0")
    if len(data) != len(spike_times[0]):
        raise ValueError('data and spike_times must have the same length')
        sys.exit("Error: data and spike_times must have the same length")
    if type(interval) != int:
        raise ValueError('interval must be an integer')
        sys.exit("Error: interval must be an integer")
    else:
        pass


def _index_data(data, spike_times, colname="X"):
    '''companion to function compute_firing_rate_data
    Inputs:
        - spike_train: the spike train of the neuron
        - spike_times: the spike_times of the spike train
    Outputs:
        - indexed_df: a data frame with the data indexed by spike_times
    '''
    indexed_df = pd.DataFrame(data=data, columns=[colname], index=spike_times[0])

    return indexed_df


def compute_firing_rates(spike_train, spike_times, interval=1):
    '''This function computes the firing rate of a given neuron.
    - Inputs: spike_train: the spike train of the neuron
              spike_times: the timestamps of the spike train
              interval: the time interval between each spike
    - Output: firing_rates: the firing rate of the neuron
    '''
    _check_inputs(spike_train, spike_times, interval)

    indexed_df = _index_data(spike_train, spike_times)

    firing_rates = []
    firing_rate_spike_times = []
    i = 1
    while i*interval <= spike_times[0][-1]+0.02:
        start_time = (i-1)*interval
        end_time = (i)*interval
        firing_rate = indexed_df[start_time:end_time].sum()/interval
        firing_rates.append(int(firing_rate))
        firing_rate_spike_times.append(start_time)
        i += 1

    firing_rate_data = pd.DataFrame(data=firing_rates, columns=['firing_rate'], index=firing_rate_spike_times)

    return firing_rate_data


def _compute_average_speeds(speeds, spike_times, interval=1):
    '''This function computes the average speed of the mice.
    Inputs:
        - speed: the speed of the mice
        - spike_times: the spike_times of the speed data
    Output:
        - average_speed: the average speed of the mice
    '''
    _check_inputs(speeds, spike_times, interval)

    indexed_df = _index_data(speeds, spike_times)

    speeds = []
    speed_spike_times = []
    i = 1
    while i*interval <= spike_times[0][-1]+0.02:
        start_time = (i-1)*interval
        end_time = (i)*interval
        speed = indexed_df[start_time:end_time].mean()
        speeds.append(int(speed))
        speed_spike_times.append(start_time)
        i += 1

    average_speeds = pd.DataFrame(data=speeds, columns=['average_speed'], index=speed_spike_times)

    return average_speeds


def plot_speed_vs_firing_rate(spike_train, speed, spike_times, figname='speed_vs_firing_rate', joint=False, showfig=False):
    '''This function plots the relationship between firing rate and average speed.
    Inputs:
        - spike_train: the spike train of the neuron
        - speed: the speed of the mice
        - spike_times: the spike_times of the speed data
    Output:
        - fig: the figure of the plot
        - conditional_mass_function: the raw conditional mass function data as a pandas dataframe
    '''

    # compute average speeds and firing rates for 1 second durations
    # and combine them into one data frame
    speed_and_rates = _compute_average_speeds(speed, spike_times, 1)
    speed_and_rates["firing_rate"] = compute_firing_rates(spike_train=spike_train, spike_times=spike_times, interval=1)

    # get sorted lists of the unique values in the data frame
    unique_speeds = sorted(speed_and_rates["average_speed"].unique())
    unique_firing_rates = sorted(speed_and_rates["firing_rate"].unique())

    # the average speeds turn out to be quantized,
    # so we filter records by each speed quantum
    # and compute the average firing rate.
    average_firing_rates = []
    for speed in unique_speeds:
        average_firing_rates.append(speed_and_rates[speed_and_rates["average_speed"] == speed]["firing_rate"].mean())
    # we also compute the max firing rate per speed quantum
    max_firing_rates = []
    for speed in unique_speeds:
        max_firing_rates.append(speed_and_rates[speed_and_rates["average_speed"] == speed]["firing_rate"].max())

    # next, we compute the mass functions of the 1 second
    # firing rates and the 1 second average speeds conditioned
    # on each speed value.
    # (i.e. the probability of firing at a certain rate
    # in a 1 second interval given an average mouse speed)

    # we start by creating a zeroed numpy array of the right size
    firing_rate_mass_function = np.zeros((len(unique_speeds), len(unique_firing_rates)))

    # we then iterate over the data frame and fill the array
    # with counts from each firing rate and speed combination
    for speed in unique_speeds:
        for firing_rate in unique_firing_rates:
            firing_rate_mass_function[speed][firing_rate] = speed_and_rates[speed_and_rates["average_speed"] == speed]["firing_rate"][speed_and_rates["firing_rate"] == firing_rate].count()

    # we then conditionally normalize the array by dividing each
    # row value by the sum of the row
    conditional_mass_function = firing_rate_mass_function/firing_rate_mass_function.sum(axis=1, keepdims=True)

    # we then format the joint and conditional mass functions
    firing_rate_mass_function = pd.DataFrame(data=firing_rate_mass_function, columns=unique_firing_rates, index=unique_speeds)
    conditional_mass_function = pd.DataFrame(data=conditional_mass_function, columns=unique_firing_rates, index=unique_speeds)

    # we then plot the conditional mass functions as a heatmap
    # and overlay the average and max firing rates as line plots
    f, ax = plt.subplots(figsize=(14, 7))
    if joint==False:
        m = ax.matshow(conditional_mass_function.T, cmap="binary")
    if joint==True:
        m = ax.matshow(firing_rate_mass_function.T, cmap="binary")
    cbar = f.colorbar(m, ax=ax, shrink=0.6)
    if joint==False:
        cbar.set_label('Firing Rate Density Conditioned On Speed', rotation=270, labelpad=20)
    if joint==True:
        cbar.set_label('Joint Density of Firing Rate and Speed', rotation=270, labelpad=20)
    ax.invert_yaxis() # flip the y axis so it is more intuitive
    plt.gca().xaxis.tick_bottom() # move the x axis ticks to the bottom
    ax.set_xticks(unique_speeds) # set the x ticks to the unique speeds
    ax.set_yticks(unique_firing_rates) # set the y ticks to unique firing rates
    ax.set_xlabel("Average Speed (??/s)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title("Mouse Speed vs Neural Firing Rate", fontsize=16, pad=20)
    ax.plot(unique_speeds, average_firing_rates, color="blue", linewidth=3)
    ax.plot(unique_speeds, max_firing_rates, color="gray", linestyle="--")
    ax.legend(["Average firing rate", "Max firing rate"], loc="upper left")

    # if showfig parameter is True, we show the figure
    if showfig==True:
        plt.show()

    plt.savefig(figname + '.png')

    return conditional_mass_function # return the conditional mass function



# def firing_rate_vs_time(spike_times: np.ndarray, pos_t: np.ndarray, window: int) -> tuple:
def firing_rate_vs_time(spike_times, pos_t, window: int) -> tuple:
    '''
        Computes firing rate as a function of time

        Params:
            times (np.ndarray):
                Array of spike_times of when the neuron fired
            pos_t (np.ndarray):
                Time array of entire experiment
            window (int):
                Defines a time window in milliseconds.

            *Example*
            window: 400 means we will attempt to collect firing data in 'bins' of
            400 millisecods before computing the firing rates.

        Returns:
            tuple: firing_rate, firing_time
            --------
            rate_vector (np.ndarray):
                Array containing firing rate data across entire experiment
            firing_time: (np.ndarray):
                spike_times of when firing occured
        '''

    # spike_times = spike_class.event_times

    # pos_t = np.array(spike_class.time_index)

    # if type(spike_times) == list:
    #     spike_times = np.array(spike_times)

    # if type(spike_times) == list:
    #     spike_times = np.asarray(spike_times)

    # Initialize zero time elapsed, zero spike events, and zero bin times.
    time_elapsed = 0
    number_of_elements = 1
    bin_time = [0,0]

    # Initialzie empty firing rate and firing time arrays
    firing_rate = [0]
    firing_time = [0]

    # Collect firing data in bins of 400ms
    for i in range(1, len(spike_times)):
        if time_elapsed == 0:
            # Set a bin start time
            bin_time_start = spike_times[i-1]

        # Increment time elapsed and spike number as spike event times are iterated over
        time_elapsed += (spike_times[i] - spike_times[i-1])
        number_of_elements += 1

        # If the elapsed time exceeds 400ms
        if time_elapsed > (window/1000):
            # Set bin end time
            bin_time_end = bin_time_start + time_elapsed
            # Compute rate, and add element to firing_rate array
            firing_rate.append(number_of_elements/time_elapsed)
            firing_time.append( (bin_time_start + bin_time_end)/2 )
            # Reset elapsed time and spiek events number
            time_elapsed = 0
            number_of_elements = 0

    rate_vector = np.zeros((len(pos_t), 1))
    index_values = []
    for i in range(len(firing_time)):
        index_values.append(  (np.abs(pos_t - firing_time[i])).argmin()  )

    firing_rate = np.array(firing_rate).reshape((len(firing_rate), 1))
    rate_vector[index_values] = firing_rate

    # spike_class.stats_dict['rate_vector'] = rate_vector

    return rate_vector, firing_time
