import numpy as np
import pandas as pd
import librosa as lb
from scipy.signal import stft

import sleep_an_tools as st

def make_avg_tfrms(subject : str, dB = True, labels=[0,1]):
    # Import the acceleration data
    accel = pd.read_csv("data/motion/{}_acceleration.txt".format(subject), sep=' ', names=['time', 'x', 'y', 'z'])
    accel = accel[accel['time'] >= 0]
    accel.reset_index(drop=True, inplace=True)
    
    # Calculate magnitude of the (x,y,z) accelerations
    accel['mag'] = st.mag(accel[['x', 'y', 'z']])
    # then calculate numerical derivative of the acceleration magnitudes.
    tstat, t0, dmag = st.vder(accel[['time', 'mag']])
    
    # Import the PSG labels
    psg = pd.read_csv("data/labels/{}_labeled_sleep.txt".format(subject), sep=' ', names=['time', 'label'])

    stages = [psg[psg['label'] == 0]['time'], psg[psg['label'] > 0]['time']]:
    
    dmag_segs = [[dmag[(t0 >= t) & (t0 <= t+30)] for t in stage] for stage in stages]
    
    stft_d = [[stft(dm, nperseg=128)[-1] for dm in dmag_segs[j] if len(dm) >= 128] for j in labels]
    
    if dB:
        stft_d = [[lb.amplitude_to_db(abs(s)) for s in stft_d[j]] for j in labels]
    else:
        stft_d = [[abs(s) for s in stft_d[j]] for j in labels]
        
    stft_d = [[np.mean(s, axis=1) for s in stft_d[j]] for j in labels]
        
    return stft_d