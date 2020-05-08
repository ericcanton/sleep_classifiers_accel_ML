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
Note that the length of the returned labels tensor need not match the input,
since we throw out labeled PSG times whose corresponding dmag windows have
too few samples (below 25 Hz) to be included. 

If called as a script, accepts the following options:
    "--len"         length of dmag windows. A positive float, optional.



    "--force" 
     or "-f"        force overwrite of existing data file



    "--split-subs"  train/test split on level of subjects (vs dmag windows)
                      * should be accompanied by "--train" or "--test"
    The following two options only relevant if "--split-subs" passed
    "--train"       size of training set. A float in (0, 1)
    "--test"        size of testing set. A float in (0, 1)
    

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
# Goal: take two lists of numpy 2-tensors, produce two numpy tensors:
#  1. first, stack of spectrograms, has shape (N, 129, q) where....
#   ... N = total number of windows in all dmags with sufficient sampling frequency
#           (determined during processing)
#   ... 129 = 1 + 128 are frequencies in the moving power spectrum. 
#   ... q = number of (2.56 = 0.02*128)-second long intervals fitting into specified window length.
#   n.b.: (129, q) is the shape of scipy.signal.stft(dmag[window], nperseg=128).
#  2. second, shape (N, 1), is PSG labels. 
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
    for d, p in zip(list_of_dmags, list_of_psgs):
        # Now iterate over times t and labels l
        for t, l in p:
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

    # Filter out those intervals we skipped and marked label as -2. 
    dmags_win = dmags_win[labels_win != -2]
    labels_win = labels_win[labels_win != -2]

    # Take STFT in segments of 128 points (2.65 seconds @ 50Hz)
    # Then boost this with amplitude_to_db, approx. 10*np.log10(../ref) but good with 0s.
    spectro_win = amplitude_to_db(np.abs(stft(dmags_win, fs=50, nperseg=128)[-1]))

    return (labels_win, dmags_win, spectro_win)

if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split as tts
    from os.path import isfile, isdir
    from os import mkdir
    from joblib import dump
    import time

    script_name = sys.argv[0]
    print(script_name, " running in script mode!")

    if "--len" in sys.argv:
        window_length = int(sys.argv[sys.argv.index("--len")+1])
    else:
        window_length = 90

#    # Fatal issue: refuse to overwrite previous outputs
#    spectro_out = "stfts_"+ str(window_len) + ".pickle."
#    label_out = "labels_"+ str(window_len) + ".pickle"
#
#    is_fatal = (isfile(spectro_out) or isfile(label_out))
#    is_fatal = is_fatal and ("-f" not in sys.argv) and ("--force" not in sys.argv)
#
#    if fatal:
#        print(spectro_out, " already exists. Move and re-run ", script_name)
#        print("Alternatively, call ", script_name, " again with -f to force.")
#        exit()
#
#    if isfile(spectro_out) and ("-f" not in sys.argv) and ("--force" not in sys.argv):
#        print(spectro_out, " already exists. Move and re-run ", script_name)
#        print("Alternatively, call ", script_name, " again with -f to force.")
#        exit()

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
    labels_win, dmags_win, spectro_win = make_stft_tensor(dmags, psg_data)
    print("(%9.4f) Tensors for neural network created..."% (now - start))

    try:
        output_dir = "../data/spectros"
        os.mkdir(output_dir)
    except:
        print("\t", output_dir, " exists...")

    if "--split-subs" in sys.argv:
        if "--train" in sys.argv:
            training_set = float(sys.argv[sys.argv.index("--train")])
        elif "--test" in sys.argv:
            training_set = 1 - float(sys.argv[sys.argv.index("--test")])
        else:
            training_set = 0.2

        labels_win_train, labels_win_test, spectro_win_train, spectro_win_test, dmags_win_test, dmags_win_train, \
                = tts(labels_win, dmags_win, spectro_win, training_size = training_set)

        files = [("labels_train.pickle", labels_win_train), \
                ("labels_test.pickle", labels_win_test), \
                ("spectro_train.pickle", spectro_win_train), \
                ("spectro_test.pickle", spectro_win_test), \
                ("dmags_test.pickle", dmags_win_test), \
                ("dmags_train.pickle", dmags_win_train)]

        
        for fname in files:
            with open(ouput_dir + fname[0], "wb") as f:
                file_start = time.time()
                dump(fname[1], f)
                now = time.time()
                print("(%9.4f) " + fname[0] + " written (took %9.4f seconds)..." % (now - start, now - file_start))

    else:
        files = [("labels.pickle", labels_win), ("dmags.pickle", dmags_win), ("spectro.pickle", spectro_win)]
        
        for fname in files:
            with open(ouput_dir + fname[0], "wb") as f:
                file_start = time.time()
                dump(fname[1], f)
                now = time.time()
                print(fname[0], " written (took %9.4f seconds)...\n" % (now - file_start))

    now = time.time()
    print("(%9.4f) Complete!\n" % (now - start))
