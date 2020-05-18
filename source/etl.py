"""
ETL scripts/functions for the sleep_classifiers_ML project.

Starts with some very short functions loading data from CSV into numpy arrays,
taking magnitude of (x,y,z) coordinates, and taking numerical derivatives. 

Then, the main data processor make_stft_tensor takes two lists
of numpy arrays: one holding the numerical derivative of acceleration magnitude (dmag),
the other holding polysomnography (PSG) labeled time windows, and returns numpy tensors:
 1. one 3-tensor holding spectrograms of windowed dmag stacked along axis 0, 
 2. one 2-tensor holding the dmag windows (after interpolation) along axis 0,
 3. one 2-tensor holding the label of the windows. 
 4. one 2-tensor holding a time stamp from the middle of each window (only if --with-time)
Note that the length of the returned labels tensor need not match the input,
since we throw out labeled PSG times whose corresponding dmag windows have
too few samples (below 25 Hz) to be included. 

If called as a script, accepts the following options:
    --len           length of dmag windows. A positive float, optional.


    --split         train/test split on level of subjects (vs dmag windows)
                      * should be accompanied by "--train" or "--test"
The following two options only relevant if "--split-subs" passed
    --train         size of training set. A float in (0, 1)
    --test          size of testing set. A float in (0, 1)

Finally, should you want to include a more explicit time component 
to the spectrogram windows, pass the flag 

    --with-time 

This outputs additional pickled numpy arrays, with shape (N,1) for spectrogram 
tensors with shape (N, 65, m). 

"""
##################################################
# Helper functions
##################################################
import glob

import pickle
import numpy as np
import pandas as pd

from scipy.signal import stft
from librosa import amplitude_to_db

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Accelerometer CSV to numpy.array (time, x, y, z)
# or PSG to numpy.array (time, label)
def load(filename : str) -> np.array:
    return pd.read_csv(filename, sep=' ').to_numpy()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (time, x, y, z) ~~> (time, |(x,y,z)|_2)
def mag(xyz : np.array) -> np.array:
    time = xyz[:,0]
    L2_mag = (xyz[:, -3]**2 + xyz[:, -2]**2 + xyz[:, -1]**2)**(0.5)
    return np.hstack((time.reshape(-1,1), L2_mag.reshape(-1,1)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (time, y) ~~> (time[:-1], y')
# where y' = (y[1:] - y[:-1])/(time[1:] - time[:-1])
def num_der(fn : np.array) -> np.array:
    y_prime = (fn[1:, 1] - fn[:-1, 1])/(fn[1:, 0] - fn[:-1, 0])
    return np.hstack((fn[1:,0].reshape(-1,1), y_prime.reshape(-1, 1)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Primary data processor ###
# Goal: take two lists of numpy 2-tensors, produce three numpy tensors:
#  1. first, stack of spectrograms, has shape (N, 65, m) where....
#   ... N = total number of windows in all dmags with sufficient sampling frequency
#           (determined during processing)
#   ... 65 = 1 + 64 are frequencies in the moving power spectrum. 
#   ... m = number of (2.56 = 0.02*128)-second long intervals fitting into specified window length.
#   n.b.: (65, m) is the shape of scipy.signal.stft(dmag[window], nperseg=128).
#  2. second, shape (N, 1), is PSG labels. 
#  3. third, shape (N, length_of_window/0.2) with rows corresponding to the window times
# 
# Basic plan: 
#   - Prepare windows
#   - "for" over zip(list_of_dmags, list_of_psgs), 
#   -- "for" over psgs
#   --- interpolate dmags in windows indexed by psg windows
#   - Take stft of dmags
def make_stft_tensor(list_of_dmags : list, list_of_psgs : list, length_of_window = 90.0) -> list:
    if len(list_of_dmags) != len(list_of_psgs):
        raise IndexError("list_of_dmags must have the same length as list_of_psgs")
        return None
    # Window length must be positive
    if length_of_window <= 0:
        raise ValueError("length_of_window must be > 0.")
        return None

    # Adjust length_of_window slightly so that 128 divides (length_of_window/0.02)
    # This becomes important when we're applying STFT to our windows, assuming sampling at 50Hz (0.02 second gaps).
    #   (More info below)
    length_of_window = 0.02*int(length_of_window/0.02) # Make length_of_window an integer multiple of 0.02
    overflow = int(length_of_window / 0.02) % 128
    length_of_window = length_of_window + 0.02*(128 - overflow) # Now (length_of_window/0.02) % 128 == 0

    # Determine the number of windows we'll be making. 
    # One window per PSG labeled 30-second window
    n_wins = sum([len(p) for p in list_of_psgs])
    win_radius = length_of_window/2

    # Set up tensors
    dmags_win = np.zeros((n_wins, int(length_of_window/0.02))) # windowed dmags
    labels_win = np.zeros(n_wins) # labels; modify with "-2" class if insufficient sampling in window
    time_window_base = np.arange(0, length_of_window, 0.02) # sliding time window
    
    # Counter for axis-0 index of tensors
    count = 0

    # Zip together the lists of dmags and psg 
    len_of_dmags = len(list_of_dmags)
    dmag_counter = 0
    for d, p in zip(list_of_dmags, list_of_psgs):
        dmag_counter += 1
        # Now iterate over times t and labels l
        N = len(p)
        inner_count = 0
        for t, l in p:
            inner_count += 1
            #print("  %d of %d is %3.2f" % (dmag_counter, len_of_dmags, 100*(inner_count/N)) + "% complete")
            # t+15 is in the middle of 30s interval
            #    t_start/end = (t+15) +/- win_radius 
            # gives a window of length (length_of_window) centered at t+15.
            t_start = (t+15) - win_radius # win_radius defaults to 45
            t_end = (t+15) + win_radius
    
            #this_d = d[(d['time'] >= t_start) & (d['time'] <= t_end)]
            this_d = d[(d[:,0] >= t_start) & (d[:,0] <= t_end)]
    
            # Now we want to determine when there are "too few" samples in a window.
            # There are some long temporal gaps in accelerometer data, and even some negative (??)..
            # ..so we consider mean to be unreliable. We use median timestep length (MTL) instead.
            # ** Sample median for MTL is 0.019952.
            # ** Sample standard deviation of MTL is 0.020213.
            # Thus, we consider unreliable any windows with < (length_of_window/0.04) samples. 
            if len(this_d) < (length_of_window/0.04):
                # Use "-2" as the class for "discarded/unreliable window"
                labels_win[count] = -2
                # Don't forget the counter!
                count += 1
                continue # Skip to next time in current PSG.
    
            time = time_window_base + t_start

            dmags_win[count, :] += np.interp(time, this_d[:,0], this_d[:,1])
            labels_win[count] = l
    
            # Another window and interpolation done. Increase the counter.
            count += 1
        print("%d of %d done..." % (dmag_counter, len_of_dmags))

    # Filter out those intervals we skipped and marked label as -2. 
    dmags_win = dmags_win[labels_win != -2]
    labels_win = labels_win[labels_win != -2]

    # Take STFT in segments of 128 points (2.65 seconds @ 50Hz)
    # Then boost this with amplitude_to_db, approx. 10*np.log10(../ref) but good with 0s.
    print("Working on spectrograms now...")
    spectro_win = amplitude_to_db(np.abs(stft(dmags_win, fs=50, nperseg=128)[-1]))

    return (labels_win, dmags_win, spectro_win)

if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split as tts
    from os.path import isfile, isdir
    from os import mkdir
    from joblib import dump
    import time

    print(sys.argv[0], " running in script mode!")

    if "--len" in sys.argv:
        window_length = int(sys.argv[sys.argv.index("--len")+1])
    else:
        window_length = 90

    # Load the data 
    sub_data = []
    psg_data = []
    count = 0
    start = time.time()
    for g in glob.glob("../data/motion/*.txt"):
        count += 1

        # data/motion/1234567_accelerometer.txt 
        #   ~~> number = 1234567_accelerometer.txt 
        #   ~~> number = 1234567
        number = g.split("/")[-1] 
        number = number.split("_")[0]
        now = time.time()
        print("(%9.4f) Working on %s (%d of 31)..." % (now - start, number, count))

        labels = number + "_labeled_sleep.txt"

        sub_data.append(load(g))
        psg_data.append(load("../data/labels/" + labels))
        now = time.time()
        print("(%9.4f) Accelerometer and PSG data loaded...\n" % (now - start))

    dmags = [num_der(mag(s)) for s in sub_data]
    now = time.time()
    print("(%9.4f) dmags calculated..."% (now - start))

    if "--split" in sys.argv:
        now = time.time()
        print("(%9.4f) Doing train/test split on subjects BEFORE spectrograms..." % (now - start))
        try:
            if "--train" in sys.argv:
                training_size = float(sys.argv[sys.argv.index("--train")+1])
            elif "--test" in sys.argv:
                training_size = 1 - float(sys.argv[sys.argv.index("--test")+1])
            else:
                training_size = 0.8
        except:
            training_size = 0.8
        print("training size = %f" % training_size)

        dmags_train, dmags_test, psg_train, psg_test = tts(dmags, psg_data, train_size=training_size)


        psg_train_win, dmags_train_win, spectro_train_win = make_stft_tensor(dmags_train, psg_train)
        now = time.time()
        print("(%9.4f) Training tensors for neural network created..."% (now - start))



        psg_test_win, dmags_test_win, spectro_test_win = make_stft_tensor(dmags_test, psg_test)
        now = time.time()
        print("(%9.4f) Testing tensors for neural network created..."% (now - start))

        now = time.time()
        print("(%9.4f) Dumping spectrogram pickles..."% (now - start))
        with open("spectros_train.pickle", "wb") as f:
            pickle.dump(spectro_train_win, f)
        with open("spectros_test.pickle", "wb") as f:
            pickle.dump(spectro_test_win, f)

        now = time.time()
        print("(%9.4f) Dumping windowed dmag pickles..."% (now - start))
        with open("dmags_train.pickle", "wb") as f:
            pickle.dump(dmags_train_win, f)
        with open("dmags_test.pickle", "wb") as f:
            pickle.dump(dmags_test_win, f)

        now = time.time()
        print("(%9.4f) Dumping PSG pickles..."% (now - start))
        with open("psg_train.pickle", "wb") as f:
            pickle.dump(psg_train_win, f)
        with open("psg_test.pickle", "wb") as f:
            pickle.dump(psg_test_win, f)

    else:
        now = time.time()
        print("(%9.4f) Calculating windows and spectrograms..." % (now - start))

        psg_win, dmags_win, spectro_win = make_stft_tensor(dmags, psg_data)
        now = time.time()
        print("(%9.4f) Tensors for neural network created..."% (now - start))

        now = time.time()
        print("(%9.4f) Dumping spectrogram pickle..."% (now - start))
        with open(output_dir + "spectros.pickle", "wb") as f:
            pickle.dump(spectro_win, f)

        now = time.time()
        print("(%9.4f) Dumping windowed dmag pickle..."% (now - start))
        with open(output_dir + "dmags.pickle", "wb") as f:
            pickle.dump(dmags_win, f)

        now = time.time()
        print("(%9.4f) Dumping PSG pickle..."% (now - start))
        with open(output_dir + "psg.pickle", "wb") as f:
            pickle.dump(psg_win, f)

    now = time.time()
    print("(%9.4f) Complete!\n" % (now - start))
