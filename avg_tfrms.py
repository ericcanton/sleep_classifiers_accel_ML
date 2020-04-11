import numpy as np
import pandas as pd
import librosa as lb
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import sleep_an_tools as st

def get_accel(subject : str, non_neg = True):
    # Import the acceleration data
    accel = pd.read_csv("data/motion/{}_acceleration.txt".format(subject), sep=' ', names=['time', 'x', 'y', 'z'])
    if non_neg:
        accel = accel[accel['time'] >= 0]
    accel.reset_index(drop=True, inplace=True)
    
    # Calculate magnitude of the (x,y,z) accelerations
    accel['mag'] = st.mag(accel[['x', 'y', 'z']])
    # then calculate numerical derivative of the acceleration magnitudes.
    tstat, t0, dmag = st.vder(accel[['time', 'mag']])

    dmag_df = pd.DataFrame({'time' : t0, 'dmag' : dmag})

    return (accel, dmag_df)

def make_avg_tfrms(subject : str, dB = True, non_neg = True, labels=[0,1]):
    # Import the acceleration data
    accel = pd.read_csv("data/motion/{}_acceleration.txt".format(subject), sep=' ', names=['time', 'x', 'y', 'z'])
    if non_neg:
        accel = accel[accel['time'] >= 0]
    accel.reset_index(drop=True, inplace=True)
    
    # Calculate magnitude of the (x,y,z) accelerations
    accel['mag'] = st.mag(accel[['x', 'y', 'z']])
    # then calculate numerical derivative of the acceleration magnitudes.
    tstat, t0, dmag = st.vder(accel[['time', 'mag']])
    
    # Import the PSG labels
    psg = pd.read_csv("data/labels/{}_labeled_sleep.txt".format(subject), sep=' ', names=['time', 'label'])

    stages = [psg[psg['label'] == 0]['time'], psg[psg['label'] > 0]['time']]
    
    dmag_segs = [[dmag[(t0 >= t) & (t0 <= t+30)] for t in stage] for stage in stages]
    
    stft_d = [[stft(dm, nperseg=128)[-1] for dm in dmag_segs[j] if len(dm) >= 128] for j in labels]
    
    if dB:
        stft_d = [[lb.amplitude_to_db(abs(s)) for s in stft_d[j]] for j in labels]
    else:
        stft_d = [[abs(s) for s in stft_d[j]] for j in labels]
        
    stft_d = [[np.mean(s, axis=1) for s in stft_d[j]] for j in labels]
        
    return stft_d

# dmag_df should have columns named 'time' and 'dmag'.
# This is the case when dmag_df comes from get_accel. 
def tfrm(dmag_df : pd.DataFrame, tstart : np.float64, tstop : np.float64, dB = True, mean = False):
    dmag_seg = dmag_df[(dmag_df['time'] >= tstart) & (dmag_df['time'] <= tstop)]['dmag'].to_numpy()
    
    stft_d = stft(dmag_seg, nperseg=128)[-1]
    
    if dB:
        stft_d = lb.amplitude_to_db(abs(stft_d))
    else:
        stft_d = abs(stft_d)
        
    if mean:
        stft_d = np.mean(stft_d, axis=1)
        stft_d.reshape(1, -1) 

    return stft_d

## This function is VERY expensive in time and space.
# -- dmag_df should have columns named 'time' and 'dmag'
# -- model should be a Stochastic Gradient Descent Classifer object trained on UN-STANDARDIZED averaged transforms.
def find_epochs(dmag_df : pd.DataFrame, model : SGDClassifier, tstart = 0, epsilon = 30):
    # This is the list of epochs to be returned. 
    # epochs[j] = [sleep stage (0 awake/1 asleep), epoch start, epoch end]
    epochs_stages = []
    epochs_times = []

    end_time = dmag_df['time'].max()

    # Stopping point; initial transform is of dmag_df over time interval [tstart, tend]
    tstop = tstart + epsilon

    ################################################################################ 
    # Compute STFT of dmag_df over time interval [tstart, tstop].
    # Then, take temporal mean of this transform, standardize that vector.
    ################################################################################ 
    running_tfrm = tfrm(dmag_df, tstart, tstop)
    #running_avg = np.mean(running_tfrm, axis=1)
    #running_avg_std = (running_avg - running_avg.mean())/running_avg.std()

    this_stage = next_stage = model.predict(np.mean(running_tfrm, axis=1).reshape(1, -1))

    while tstop < end_time:
        while this_stage == next_stage:
            # Possibly this epoch stretches the remainder of the time...
            if tstop < end_time:
                tstop += epsilon
            # ...if so, break to complete the process.
            else:
                break

            # Extend the transform, compute new class. 
            running_tfrm = np.concatenate((running_tfrm, tfrm(dmag_df, tstart, tstop)), axis=1)
            #running_avg = np.mean(running_tfrm, axis=1)
            #running_avg_std = (running_avg - running_avg.mean())/running_avg.std()
            next_stage = model.predict(np.mean(running_tfrm, axis=1).reshape(1, -1))

            # If next_stage != this_stage, this while exits. 
            # Then, we should record our epoch, prepare to find next one.

        # If running this code, either next_stage changed or tstop >= end_time.
        # In either case, we want to record the epoch we just found.
        epochs_stages += [this_stage]
        epochs_times += [tstart]

        # Reset interval to [tstop-epsilon, tstop]. 
        tstart = tstop - epsilon

    return pd.DataFrame({'time' : epochs_times, 'label' : epochs_stages})
