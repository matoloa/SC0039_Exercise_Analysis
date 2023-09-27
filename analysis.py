# %%
import os
from pathlib import Path
import numpy as np # numeric calculations module
import pandas as pd # dataframe module
import scipy # peakfinder and other useful analysis tools

'''
# NOTE for Python course assignment:
The standalone test run, which is the aspect to be evaluated in this course, starts on 283.
Focus on the functions called after the main-guard: find_all_t and build_df_result.
'''


# %%
def build_df_result(df_data, t_EPSP_amp):#, t_EPSP_slope, t_EPSP_slope_size, t_volley_amp, t_volley_slope, t_volley_slope_size, output_path):
    # Incomplete function: only resolves EPSP_amp for now
    """Measures each sweep in df (e.g. from <save_file_name>.csv) at specificed times t_*
    Args:
        df: a dataframe containing numbered sweeps, timestamps and voltage
        t_EPSP_amp: time of lowest point of EPSP
        t_EPSP_slope: time of centre of EPSP_slope
        t_EPSP_slope_size: width of EPSP slope
        t_volley_amp: time of lowest point of volley
        t_volley_slope: time of centre of volley_slope
        t_volley_slope_size: width of volley slope
        Optional
        output_path: if present, store results to this path (csv)
    Returns:
      a dataframe. Per sweep (row): EPSP_amp, EPSP_slope, volley_amp, volley_EPSP
    """
    df_result = pd.DataFrame()
    df_result['sweep'] = df_data.sweep.unique() # one row per unique sweep in data file
    df_EPSP_amp = df_data[df_data['time']==t_EPSP_amp].copy() # filter out all time (from sweep start) that do not match t_EPSP_amp
    df_EPSP_amp.reset_index(inplace=True)
    df_result['EPSP_amp'] = df_EPSP_amp['voltage'] # add the voltage of selected times to df_result
    return df_result


# %%
def find_i_stim_prim_max(dfmean): # steepest incline = stimulation artefact
    # TODO: return an index of sufficiently separated over-threshold x:es instead
    return dfmean["prim"].idxmax()


# %%
def find_i_EPSP_peak_max(
    dfmean,
    limitleft=0,
    limitright=-1,
    param_EPSP_minimum_width_ms=5, # width in ms
    param_EPSP_minimum_prominence_mV=0.001, # what unit? TODO: find out!
    ):
    """
    width and limits in index, promincence in Volt
    returns index of center of broadest negative peak on dfmean
    """
    print("find_i_EPSP_peak_max:")

    # calculate sampling frequency
    time_delta = dfmean.time[1] - dfmean.time[0]
    sampling_Hz = 1 / time_delta
    print(f" . . . sampling_Hz: {sampling_Hz}")

    # convert EPSP width from ms to index
    EPSP_minimum_width_i = int(
    param_EPSP_minimum_width_ms * 0.001 * sampling_Hz
    ) # 0.001 for ms to seconds, *sampling_Hz for seconds to recorded points
    print(" . . . EPSP is at least", EPSP_minimum_width_i, "points wide")

    # scipy.signal.find_peaks returns a tuple
    i_peaks, properties = scipy.signal.find_peaks(
        -dfmean["voltage"],
        width=EPSP_minimum_width_i,
        prominence=param_EPSP_minimum_prominence_mV / 1000,
        )
    print(f" . . . i_peaks:{i_peaks}")
    if len(i_peaks) == 0:
        print(" . . No peaks in specified interval.")
        return np.nan
    print(f" . . . properties:{properties}")

    dfpeaks = dfmean.iloc[i_peaks]
    # dfpeaks = pd.DataFrame(peaks[0]) # Convert to dataframe in order to select only > limitleft
    dfpeaks = dfpeaks[limitleft < dfpeaks.index]
    print(f" . . . dfpeaks:{dfpeaks}")

    i_EPSP = i_peaks[properties["prominences"].argmax()]
    return i_EPSP


# %%
def find_i_VEB_prim_peak_max(
    dfmean,
    i_Stim,
    i_EPSP,
    param_minimum_width_of_EPSP=5, # ms
    param_minimum_width_of_VEB=1, # ms
    param_prim_prominence=0.0001, # TODO: correct unit for prim?
    ):
    """
    returns index for VEB (Volley-EPSP Bump - the notch between volley and EPSP)
    defined as largest positive peak in first order derivative between i_stim and i_EPSP
    """
    print("find_i_VEB_prim_peak_max:")
    # calculate sampling frequency
    time_delta = dfmean.time[1] - dfmean.time[0]
    sampling_Hz = 1 / time_delta
    print(f" . . . sampling_Hz: {sampling_Hz}")

    # convert time constraints (where to look for the VEB) to indexes
    minimum_acceptable_i_for_VEB = int(i_Stim + 0.001 * sampling_Hz) # The VEB is not within a ms of the i_stim
    max_acceptable_i_for_VEB = int(
    i_EPSP - np.floor((param_minimum_width_of_EPSP * 0.001 * sampling_Hz) / 2)
    ) # 0.001 for ms to seconds, *sampling_Hz for seconds to recorded points
    print(" . . . VEB is between", minimum_acceptable_i_for_VEB, "and", max_acceptable_i_for_VEB)

    # create a window to the acceptable range:
    prim_sample = dfmean["prim"].values[minimum_acceptable_i_for_VEB:max_acceptable_i_for_VEB]

    # find the sufficiently wide and promintent peaks within this range
    i_peaks, properties = scipy.signal.find_peaks(
    prim_sample,
    width=param_minimum_width_of_VEB * 1000 / sampling_Hz, # *1000 for ms to seconds, / sampling_Hz for seconds to recorded points
    prominence=param_prim_prominence / 1000, # TODO: unit?
    )

    # add skipped range to found indexes
    i_peaks += minimum_acceptable_i_for_VEB
    print(" . . . i_peaks:", i_peaks)
    if len(i_peaks) == 0:
        print(" . . No peaks in specified interval.")
        return np.nan
    print(f" . . . properties:{properties}")
    i_VEB = i_peaks[properties["prominences"].argmax()]
    print(f" . . . i_VEB: {i_VEB}")
    return i_VEB


# %%
def find_all_i(dfmean, param_min_time_from_i_Stim=0.0005, verbose=False):
    """
    runs all index-detections in the appropriate sequence,
    The function finds VEB, but does not currently report it
    TODO: also report volley amp and slope
    Returns a dict of all DETECTED indices: point for amp(litudes), center for slopes.
    """

    i_Stim = np.nan
    i_EPSP_amp = np.nan
    i_VEB = np.nan
    i_Stim = find_i_stim_prim_max(dfmean)
    if verbose:
        print(f"i_Stim:{i_Stim}")
    if i_Stim is not np.nan:
        i_EPSP_amp = find_i_EPSP_peak_max(dfmean)
        if verbose:
            print(f"i_EPSP_amp:{i_EPSP_amp}")
        if i_EPSP_amp is not np.nan:
            i_VEB = find_i_VEB_prim_peak_max(dfmean, i_Stim, i_EPSP_amp)
            if verbose:
                print(f"i_VEB:{i_VEB}")
    return {"i_Stim": i_Stim, "i_VEB": i_VEB, "i_EPSP_amp": i_EPSP_amp}


# %%
def find_all_t(dfmean, param_min_time_from_i_Stim=0.0005, verbose=False):
    """
    Acquires indices via find_all_t() for the provided dfmean and converts them to time values
    Returns a dict of all t-values provided by find_all_t()
    """
    if verbose:
       print("find_all_t")
    dict_i = find_all_i(dfmean, param_min_time_from_i_Stim=0.0005, verbose=verbose)
    dict_t = {}
    for k, v in dict_i.items():
       k_new = "t" + k[1:]
       dict_t[k_new] = np.nan if v is np.nan else dfmean.loc[v].time
    if verbose:
      print(f"dict_t: {dict_t}")
    return dict_t


# %%
''' Standalone test:'''
if __name__ == "__main__":
    for _ in range(3): # add some space
        print()
    print("Running as main: standalone test")
    print()

    cwd_source = os.getcwd() + "/source" # locate supplied sample files
    path_datafile = Path(cwd_source + "/data.csv")
    path_meanfile = Path(cwd_source + "/mean.csv")
    try:
        df_data = pd.read_csv(str(path_datafile)) # a persisted csv-form of the data file
        df_mean = pd.read_csv(str(path_meanfile)) # a persisted average of all sweeps in that data file
    except FileNotFoundError:
        print(f"For this standalone test to work, the source files must be in the folder {cwd_source}. Sorry about that.")

    # adding calibrated voltage to df_data (in later iterations, this will be done upon parsing the raw data)
    dfpivot = df_data[['sweep', 'voltage', 'time']].pivot_table(values='voltage', columns = 'time', index = 'sweep')
    ser_startmedian = dfpivot.iloc[:,:20].median(axis=1)
    df_calibrated = dfpivot.subtract(ser_startmedian, axis = 'rows')
    df_calibrated = df_calibrated.stack().reset_index()
    df_calibrated.rename(columns = {0: 'voltage'}, inplace=True)
    df_calibrated.sort_values(by=['sweep', 'time'], inplace=True)
    df_data.rename(columns = {'voltage': 'voltage_raw'}, inplace=True)
    df_data['voltage'] = df_calibrated.voltage

    all_t = find_all_t(df_mean) # use the average all sweeps to determine where all events are located (noise reduction)
    t_EPSP_amp = all_t['t_EPSP_amp'] # use coordinates from the average on all sweeps in the data file (measure actual data)
    df_result = build_df_result(df_data=df_data, t_EPSP_amp=t_EPSP_amp)
    print()
    print("df_result:")
    print(df_result)