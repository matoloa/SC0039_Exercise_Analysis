import numpy as np  # numeric calculations module
import pandas as pd  # dataframe module, think excel, but good
import scipy  # peakfinder and other useful analysis tools
import os
from pathlib import Path

'''
    1. Finder functions: finds indices of events in data
    2. Result functions: generates results from data at indices
    3. Convenience functions
    4. Mainguard - standalone test
'''



# 1. Finder functions

def find_i_stim_prim_max(dfmean): # steepest incline = stimulation artefact
    # TODO: return an index of sufficiently separated over-threshold x:es instead
    return dfmean["prim"].idxmax()


def find_i_EPSP_peak_max(
    dfmean,
    limitleft=0,
    limitright=-1,
    param_EPSP_minimum_width_ms=5, # width in ms
    param_EPSP_minimum_prominence_mV=0.001, # smallest acceptable prominence in mV
    ):
    """
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
    print(f" . . . i_peaks: {i_peaks}")
    if len(i_peaks) == 0:
        print(" . . No peaks in specified interval.")
        return np.nan

    dfpeaks = dfmean.iloc[i_peaks]
    # dfpeaks = pd.DataFrame(peaks[0]) # Convert to dataframe in order to select only > limitleft
    dfpeaks = dfpeaks[limitleft < dfpeaks.index]
    i_EPSP = i_peaks[properties["prominences"].argmax()]
    return i_EPSP


def find_i_VEB_prim_peak_max(
    dfmean,
    i_stim,
    i_EPSP,
    param_minimum_width_of_EPSP=5, # ms
    param_minimum_width_of_VEB=1, # ms
    param_prim_prominence=0.0001, # TODO: use correct unit for prim
    ):
    """
    returns index for VEB (Volley-EPSP Bump - the notch between volley and EPSP)
    defined as largest positive peak in first order derivative between i_stim and i_EPSP
    """
    print("find_i_VEB_prim_peak_max:")
    # calculate sampling frequency (again, in case it's called without find_i_EPSP_peak_max)
    time_delta = dfmean.time[1] - dfmean.time[0]
    sampling_Hz = 1 / time_delta
    print(f" . . . sampling_Hz: {sampling_Hz}")
    # convert time constraints (where to look for the VEB) to indexes
    minimum_acceptable_i_for_VEB = int(i_stim + 0.001 * sampling_Hz) # The VEB is not within a ms of the i_stim
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
    i_VEB = i_peaks[properties["prominences"].argmax()]
    print(f" . . . i_VEB: {i_VEB}")
    return i_VEB


def find_i_EPSP_slope(dfmean, i_VEB, i_EPSP):
    """
    Find index of the point where voltage bis crosses over zero (from negative to positive); "the straigthest line".
    """
    dftemp = dfmean.bis[i_VEB:i_EPSP]
    i_EPSP_slope = dftemp[0 < dftemp.apply(np.sign).diff()].index.values

    if len(i_EPSP_slope) == 0:
        print(" . . No positive zero-crossings in dfmean.bis[i_VEB: i_EPSP].")
        return np.nan
    return i_EPSP_slope[0]


def find_all_i(dfmean, param_min_time_from_i_stim=0.0005):
    """
    runs all index-detections in the appropriate sequence,
    The function finds VEB, but does not currently report it
    TODO: also report volley amp and slope
    Returns a dict of all indices, with np.nan representing detection failure.
    """
    dict_i = { #set default np.nan
        "i_stim": np.nan,
        "i_VEB": np.nan,
        "i_EPSP_amp": np.nan,
        "i_EPSP_slope": np.nan}
    dict_i['i_stim'] = find_i_stim_prim_max(dfmean=dfmean,)
    if dict_i['i_stim'] is np.nan: # TODO: will not happen in current configuration
        return dict_i
    dict_i['i_EPSP_amp'] = find_i_EPSP_peak_max(dfmean=dfmean)
    if dict_i['i_EPSP_amp'] is np.nan: # TODO: will not happen in current configuration
        return dict_i
    dict_i['i_VEB'] = find_i_VEB_prim_peak_max(dfmean=dfmean, i_stim=dict_i['i_stim'], i_EPSP=dict_i['i_EPSP_amp'])
    if dict_i['i_VEB'] is np.nan:
        return dict_i
    dict_i['i_EPSP_slope'] = find_i_EPSP_slope(dfmean=dfmean, i_VEB=dict_i['i_VEB'] , i_EPSP=dict_i['i_EPSP_amp'])
    return dict_i


def find_all_t(dfmean, param_min_time_from_i_stim=0.0005):
    """
    Acquires indices via find_all_t() for the provided dfmean and converts them to time values
    Returns a dict of all t-values provided by find_all_t()
    """
    dict_i = find_all_i(dfmean, param_min_time_from_i_stim=0.0005)
    dict_t = i2t(dfmean, dict_i)
    print(f"dict_t: {dict_t}")
    return dict_t



# 2. Result functions

def build_dfoutput(dfdata, t_EPSP_amp=None, t_EPSP_slope=None):
    """Measures each sweep in df (e.g. from <save_file_name>.csv) at specificed times t_* 
    Args:
        df: a dataframe containing numbered sweeps and voltage
        t_EPSP_amp: time of lowest point of EPSP
        t_EPSP_slope: time of centre of EPSP_slope
    Returns:
        a dataframe. Per sweep (row): EPSP_amp, EPSP_slope
    """
    dfoutput = pd.DataFrame()
    dfoutput['sweep'] = dfdata.sweep.unique() # one row per unique sweep in data file
    # EPSP_amp
    if t_EPSP_amp is not None:
        if t_EPSP_amp is not np.nan:
            df_EPSP_amp = dfdata[dfdata['time']==t_EPSP_amp].copy() # filter out all time (from sweep start) that do not match t_EPSP_amp
            df_EPSP_amp.reset_index(inplace=True)
            dfoutput['EPSP_amp'] = df_EPSP_amp['voltage'] # add the voltage of selected times to dfoutput
        else:
            dfoutput['EPSP_amp'] = np.nan
    # EPSP_slope
    if t_EPSP_slope is not None:
        if t_EPSP_slope is not np.nan:
            df_EPSP_slope = measureslope_vec(dfdata=dfdata, t_slope=t_EPSP_slope, halfwidth=0.0004)
            dfoutput['EPSP_slope'] = df_EPSP_slope['value']
        else:
            dfoutput['EPSP_slope'] = np.nan
    return dfoutput


def measureslope_vec(dfdata, t_slope, halfwidth, name="EPSP"):
    """
    vectorized measure slope; returns a dataframe with one row per sweep and <name>_slope column
    """
    # set relevant window to dfdata
    df = dfdata[((t_slope - halfwidth) <= dfdata.time) & (dfdata.time <= (t_slope + halfwidth))]
    # pivot to get one row per sweep
    dfpivot = df.pivot(index='sweep', columns='time', values='voltage')
    # calculate slope for each sweep
    coefs = np.polyfit(dfpivot.columns, dfpivot.T, deg=1).T
    dfslopes = pd.DataFrame(index=dfpivot.index)
    # add slope to dataframe
    dfslopes['type'] = name + "_slope"
    dfslopes['algorithm'] = 'linear'
    dfslopes['value'] = coefs[:, 0]  # TODO: verify that it was the correct columns, and that values are reasonable
    return dfslopes



# 3. Convenience functions

def i2t(dfmean, dict_i):
    # Converts dict_i (index) to dict_t (time from start of sweep in dfmean)
    dict_t = {}
    for k, v in dict_i.items():
        k_new = "t" + k[1:]
        dict_t[k_new] = np.nan if v is np.nan else dfmean.loc[v].time
    return dict_t



# 4. Mainguard - Standalone test
if __name__ == "__main__":
    for _ in range(3): # add some space
        print()
    print("Running as main: standalone test")
    print()
    cwd_source = os.getcwd() + "/source" # locate supplied sample files
    path_datafile = Path(cwd_source + "/data.csv")
    path_meanfile = Path(cwd_source + "/mean.csv")
    try:
        dfdata = pd.read_csv(str(path_datafile)) # a persisted csv-form of the data file
        dfmean = pd.read_csv(str(path_meanfile)) # a persisted average of all sweeps in that data file
    except FileNotFoundError:
        print(f"For this standalone test to work, the source files must be in the folder {cwd_source}. Sorry about that.")

    dict_t = find_all_t(dfmean) # use the average all sweeps to determine where all events are located (noise reduction)
    t_EPSP_amp = dict_t['t_EPSP_amp']
    t_EPSP_slope = dict_t['t_EPSP_slope']
    dfoutput = build_dfoutput(dfdata=dfdata,
                              t_EPSP_amp=t_EPSP_amp,
                              t_EPSP_slope=t_EPSP_slope)
    print(dfoutput)