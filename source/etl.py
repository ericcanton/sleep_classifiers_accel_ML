"""
ETL and feature extraction functions for the sleep_classifiers_ML project.

Starts with some very short functions loading data from CSV into numpy arrays,
taking magnitude of (x,y,z) coordinates, and taking numerical derivatives. 

Then, the main data processor make_stft_tensor takes two lists
of numpy arrays: one holding the numerical derivative of acceleration magnitude (dmag),
the other holding polysomnography (PSG) labeled time windows, and returns numpy tensors:
 1. one 3-tensor holding spectrograms of windowed dmag stacked along axis 0, 
 2. one 2-tensor holding the dmag windows (after interpolation) along axis 0,
 3. one 2-tensor holding the label of the windows. 
 4. one 2-tensor holding a time stamp from the middle of each window (only if --with-time)
Note that the axis-0 length of the returned tensors need be easily be 
calculated/predicted based on the number of scored PSG epochs:
we throw out labeled PSG windows whose corresponding dmag windows have
too few samples (below 25 Hz) to be included. 

If called as a script, accepts the following options:
    --len           length of dmag windows. A positive float, optional.
    --sep           do each subject separately, store in folders in ../data/pickles/
    --split         train/test split on level of subjects (vs dmag windows)
The following two options only relevant if "--split-subs" passed
    --train         size of training set. A float in (0, 1)
    --test          size of testing set. A float in (0, 1)

Finally, should you want to include a more explicit time component 
to the spectrogram windows, pass the flag 

    --with-time 

This outputs additional pickled numpy arrays, with shape (N,1) for spectrogram 
tensors with shape (N, 65, *). 

"""
import glob

import pickle
import numpy as np
import pandas as pd

from scipy.signal import stft
from librosa import amplitude_to_db
import sys
from sklearn.model_selection import train_test_split
from os.path import isfile, isdir
import os
import time


##################################################
# Helper functions
##################################################
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

##################################################
### Primary data processor 
##################################################
"""
 Goal: take two lists of numpy 2-tensors, produce three or four numpy tensors:
  1. axis-0 stack of spectrograms, shape (N, 65, m), where....
   ... N = total number of windows in all dmags with sufficient sampling frequency
           (determined during processing)
   ... 65 = 1 + 64 are frequencies in the moving power spectrum. 
   ... m = number of (2.56 = 0.02*128)-second long intervals fitting into specified window length.
   n.b.: (65, m) is the shape of scipy.signal.stft(dmag[window], nperseg=128).
  2. PSG label for time at midpoint of window, shape (N, 1).
  3. axis-0 stack of dmag windows, shape (N, length_of_window/0.02) [approximately**]
  4. window start/mid/end times, shape (N,3), corresponding to the spectrogram
    with matching axis-0 index. Only if with_time = True

 ** we adjust length_of_window so that 128 divides length_of_window/0.02
 
 Basic plan: 
   - Prepare windows; adjust length so 128 divides the number of sample times 
   - "for" over zip(list_of_dmags, list_of_psgs), 
   -- "for" over psgs
   --- interpolate dmags in windows indexed by psg windows
   - Take stft of dmags
"""
def make_stft_tensor(list_of_dmags : list, list_of_psgs : list, length_of_window = 90.0, with_time = True) -> tuple:
    print("Working inside make_stft_tensor now...")
    if len(list_of_dmags) != len(list_of_psgs):
        raise IndexError("list_of_dmags and list_of_psgs must have the same length.\nReturning None!")
    # Window length must be positive
    if length_of_window <= 0:
        raise ValueError("length_of_window must be > 0.\nReturning None!")

    
    ##################################################
    ### Prepare windows
    ##################################################
    # Adjust length_of_window slightly so that 128 divides (length_of_window/0.02)
    # This becomes important when we're applying STFT to our windows, assuming sampling at 50Hz (0.02 second gaps).
    #   (More info below)
    length_of_window = 0.02*int(length_of_window/0.02) # Make length_of_window an integer multiple of 0.02
#    print("length_of_window", length_of_window)
    
    overflow = int(length_of_window / 0.02) % 128
#    print("overflow", overflow)
    
    length_of_window = length_of_window + 0.02*(128 - overflow) # Now (length_of_window/0.02) % 128 == 0
#    print("length_of_window", length_of_window)
    win_radius = length_of_window/2

    # Determine the number of windows we'll be making per-person. 
    # One window per PSG labeled 30-second window
    n_wins = sum([len(p) for p in list_of_psgs])
#    print("n_wins", n_wins)

    ##################################################
    ### Set up output tensors
    ##################################################
    dmags_win = np.zeros((n_wins, int(length_of_window/0.02))) # windowed dmags
#    print("dmags_win.shape", dmags_win.shape)
    
    labels_win = np.zeros(n_wins) # labels; modify with "-2" class if insufficient sampling in window
#    print("labels_win.shape", labels_win.shape)
    
    time_window_base = np.arange(0, length_of_window, 0.02) # sliding time window
#    print("time_window_base", time_window_base)
    
    if with_time:
        time_win = np.zeros(n_wins)
#        print("time_win.shape", time_win.shape)


    """
    ##################################################
    ### Window calculation/stacking loop
    ##################################################
    """
    dmag_progress_counter = 0 # Purely cosmetic. Used for printing updates on progress.
    len_of_dmags = len(list_of_dmags) # Also for progress reporting only
#    print("len_of_dmags", len_of_dmags)

    # count is used for the primary index of
    #   dmag_win
    #   labels_win
    #   time_win
    # Because it must increase sequentially over all elements of
    # each PSG array, we control it manually (e.g. cannot use enumerate(...) in the for loops)
    count = 0

    for d, p in zip(list_of_dmags, list_of_psgs):

#        print("dmags_win.shape", dmags_win.shape)
#        print("time_window_base", time_window_base)
#        print("labels_win.shape", labels_win.shape)

        dmag_progress_counter += 1 # Purely cosmetic. Used for printing updates on progress.

        # Now iterate over times t and labels l
        # p is a numpy array of shape (*, 2) with each row being a (time, label) pair
        for t, l in p:

            """
            ##################################################
            ### Get the window of dmag referenced by PSG
            ##################################################
            t+15 is in the middle of 30s interval
               t_start/_end = (t+15) +/- win_radius 
            gives a window of length (length_of_window) centered at t+15.
            """
            t_start = (t+15) - win_radius # win_radius defaults to 45
            t_end = (t+15) + win_radius
    
            this_d = d[(d[:,0] >= t_start) & (d[:,0] <= t_end)]
    
            """
            # Now we want to determine when there are "too few" samples in a window.
            # There are some long temporal gaps in accelerometer data, and even some negative (??)..
            # ..so we consider mean to be unreliable. We use median timestep length (MTL) instead.
            # ** Sample median for MTL is 0.019952.
            # ** Sample standard deviation of MTL is 0.020213.
            # Thus, we consider unreliable any windows with < (length_of_window/0.04) samples. 
            """
            if len(this_d) < (length_of_window/0.04):
                # Use "-2" as the class for "discarded/unreliable window"
                labels_win[count] = -2
                # Don't forget the counter!
                count += 1
                continue # Skip to next time in current PSG.
    
            time = time_window_base + t_start

            dmags_win[count, :] = np.interp(time, this_d[:,0], this_d[:,1])
#            print("dmags_win.shape", dmags_win.shape)
#            print("labels_win.shape", labels_win.shape)
#            print("count", count)
#            print("l", l)
#            print("time_window_base.shape", time_window_base.shape)
            labels_win[count] = l
#            print(labels_win[count])

            if with_time:
                time_win[count] = t+15
    
            # Another window and interpolation done. Increase the counter.
            count += 1

        # back to:       for d, p in zip(list_of_dmags, list_of_psgs)
        print("Magnitude differential windowing: %d of %d done..." % (dmag_progress_counter, len_of_dmags))

    # Filter out those intervals with labels -1 (unscored) or -2 (those we skipped)
    keep = (labels_win >= 0)
    dmags_win = dmags_win[keep]
    labels_win = labels_win[keep]
    if with_time:
        time_win = time_win[keep]

    # Take STFT in segments of 128 points (2.56 seconds @ 50Hz)
    # Then boost this with amplitude_to_db, approx. 10*np.log10(../ref) but good with 0s.
    print("Working on spectrograms now...")
#    print("dmags_win.shape", dmags_win.shape)
    spectro_win = amplitude_to_db(np.abs(stft(dmags_win, fs=50, nperseg=128)[-1]))

    # Return in the same order as described in the docstring for this method.
    if with_time:
        return spectro_win, labels_win, dmags_win, time_win
    else:
        return spectro_win, labels_win, dmags_win

if __name__ == "__main__":
    # If the user just wants "--help" or "-h" print some and exit!
    if ("--help" in sys.argv) or ("-h" in sys.argv):
        print("""
Arguments:
  --len         specify window length
  --split       train/test split on subjects, default 80/20 (24 test/7 train)
  --sep         make separate pickles for each subject in the study
  --with-time   output also start/mid/end times of the windows
  --help, -h    output this help message and exit

 If --split is passed, we have two other options (otherwise ignored)
  --train 0.x   float in [0, 1], size of training split, default 0.8
  --test 0.x    same as train, but specify test size instead, default 0.2
""")
        exit()




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
    names = []
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
        names.append(number)
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
        print("training size = %3.2f" % training_size)

        dmags_train, dmags_test, psg_train, psg_test = train_test_split(dmags, psg_data, train_size=training_size)
        now = time.time()
        print("(%9.4f) Sample splits drawn..." % (now - start))


        ##################################################
        ### Training split
        ##################################################
        if "--with-time" in sys.argv:
            with_time = True
        else:
            with_time = False
        spectro_train_win, psg_train_win, dmags_train_win, time_train_win = make_stft_tensor(dmags_train, psg_train)
        now = time.time()
        print("(%9.4f) Training tensors for neural network created..."% (now - start))



        ##################################################
        ### Testing split
        ##################################################
        spectro_test_win, psg_test_win, dmags_test_win, time_test_win = make_stft_tensor(dmags_test, psg_test, with_time = True)
        now = time.time()
        print("(%9.4f) Testing tensors for neural network created..."% (now - start))

    
        ##################################################
        ### Write the arrays to disk
        ##################################################
        now = time.time()
        print("(%9.4f) Dumping spectrogram pickles..."% (now - start))
        with open("spectros_train.pickle", "wb") as f:
            pickle.dump(spectro_train_win, f)
        with open("spectros_test.pickle", "wb") as f:
            pickle.dump(spectro_test_win, f)

        now = time.time()
        print("(%9.4f) Dumping PSG pickles..."% (now - start))
        with open("psg_train.pickle", "wb") as f:
            pickle.dump(psg_train_win, f)
        with open("psg_test.pickle", "wb") as f:
            pickle.dump(psg_test_win, f)

        now = time.time()
        print("(%9.4f) Dumping windowed dmag pickles..."% (now - start))
        with open("dmags_train.pickle", "wb") as f:
            pickle.dump(dmags_train_win, f)
        with open("dmags_test.pickle", "wb") as f:
            pickle.dump(dmags_test_win, f)


        if "--with-time" in sys.argv:
            now = time.time()
            print("(%9.4f) Dumping time pickles..."% (now - start))
            with open("time_train.pickle", "wb") as f:
                pickle.dump(time_train_win, f)
            with open("time_test.pickle", "wb") as f:
                pickle.dump(time_test_win, f)

    
    ##################################################
    ### Same as the "--split" case, but simpler
    ##################################################
    elif "--sep" in sys.argv:
        failed = []
        if "--with-time" in sys.argv:
            with_time = True
        else:
            with_time = False

        for i, n in enumerate(names):
            output_dir = "../data/pickles/%s/" % n

            if not isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            now = time.time()
            print("(%9.4f) Calculating windows and spectrograms..." % (now - start))
            try:
                spectro_win, psg_win, dmags_win, time_win = make_stft_tensor([dmags[i]], [psg_data[i]], with_time = with_time)
            except:
                print("There was an issue with %s. Continuing." % n)
                failed.append(n)
                continue

            now = time.time()
            print("(%9.4f) Tensors for neural network created..."% (now - start))

            now = time.time()
            print("(%9.4f) Dumping spectrogram pickle for %s..."% (now - start, n))
            with open(output_dir + "spectros.pickle", "wb") as f:
                pickle.dump(spectro_win, f)

            now = time.time()
            print("(%9.4f) Dumping PSG pickle for %s..."% (now - start, n))
            with open(output_dir + "psg.pickle", "wb") as f:
                pickle.dump(psg_win, f)
    
            if "--with-dmags" in sys.argv:
                now = time.time()
                print("(%9.4f) Dumping windowed dmag pickle for %s..."% (now - start, n))
                with open(output_dir + "dmags.pickle", "wb") as f:
                    pickle.dump(dmags_win, f)
    
            if with_time:
                now = time.time()
                print("(%9.4f) Dumping time pickle for %s..."% (now - start, n))
                with open(output_dir + "time.pickle", "wb") as f:
                    pickle.dump(time_win, f)

        if len(failed) > 0:
            print("We failed to compute the following subjects' spectrograms:\n", failed)
        else:
            print("All spectrograms created successfully!")

    else:
        now = time.time()
        print("(%9.4f) Calculating windows and spectrograms..." % (now - start))

        if "--with-time" in sys.argv:
            dmags_win, psg_win, spectro_win, time_win = make_stft_tensor(dmags, psg_data, with_time = True)
        else:
            dmags_win, psg_win, spectro_win = make_stft_tensor(dmags, psg_data)
        now = time.time()
        print("(%9.4f) Tensors for neural network created..."% (now - start))

        now = time.time()
        print("(%9.4f) Dumping spectrogram pickle..."% (now - start))
        with open( "spectros.pickle", "wb") as f:
            pickle.dump(spectro_win, f)

        now = time.time()
        print("(%9.4f) Dumping PSG pickle..."% (now - start))
        with open( "psg.pickle", "wb") as f:
            pickle.dump(psg_win, f)

        now = time.time()
        print("(%9.4f) Dumping windowed dmag pickle..."% (now - start))
        with open( "dmags.pickle", "wb") as f:
            pickle.dump(dmags_win, f)

        if "--with-time" in sys.argv:
            now = time.time()
            print("(%9.4f) Dumping time pickle..."% (now - start))
            with open( "time.pickle", "wb") as f:
                pickle.dump(time_win, f)


    now = time.time()
    print("(%9.4f) Complete!\n" % (now - start))
