""" 
Evaluation scripts. There are four functions defined here:
    1. logits_to_proba
    2. data_yielder
    3. split_yielder
    4. pr_roc_from_path
    5. sample_pics
    6. get_probabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import tensorflow as tf

from sklearn.metrics import roc_curve, precision_recall_curve, auc

import os
import pickle

def logits_to_proba(logit):
    """
    Parameters: logit, logarithmic odds of probability p.
        By definition, logits are:
          logit(p) = log(p/(1-p)).
        logits are easily transformed into a probability:
          p = exp(logit)/(1 + exp(logit)).
        Logits are useful when training classifier models because they do not 
        suffer from some of the numerical issues that the sigmoid function 
        does. A key issue is saturation:  if \sigma(x) is the sigmoid, then 
          (d/dx)\sigma(x) is very small for |x| >>0, 
        leading to numerical underflow that prevents gradient descent methods 
        from being effective in training classifiers. 
    
    Values: probability of given logit
        p(logit) = exp(logit)/(1+exp(logit))
    """    
    return np.exp(logit)/(1+np.exp(logit))

def sleep_classes(inputs, n_classes):
    """
    Convert 6-stage classification to n_classes.
    The six stages are:
        0: awake
        1: nrem1
        2: nrem2
        3: nrem3
        4: nrem4
        5: rem

    For our purposes, only makes sense to group these as follows:
        2 stage: sleep 0, wake 1
        3 stage: sleep 0, nrem 1, rem 2
        4 stage: sleep 0, light nrem 1, deep nrem 2, rem 3
    """
    if n_classes == 2:
        return np.where(inputs > 0, 1, 0)
    elif n_classes == 3:
        return np.piecewise(inputs, [inputs == 0, 
                                     (inputs > 0) & (inputs < 5),
                                     inputs == 5], [0, 1, 2])
    elif n_classes == 4:
        return np.piecewise(inputs, [inputs == 0, 
                                     (inputs > 0) & (inputs < 3),
                                     (inputs >= 3) & (inputs < 5),
                                     inputs == 5], [0, 1, 2, 3])
    else:
        raise ValueError("n_classes must be 2, 3 or 4.")

def data_yielder(data_path, neural_path = None, exclude = None, nn = True, nn_base_name = None):
    """
     Generator to create leave-one-out train/test splits.
     Upon initialization, loads the separate data pickles.
     Yields: (subject_number[i], spectrogram[i], time[i], psg labels[i], neural network[i])
        If exclude != None, then it's a list of subject numbers not to be yielded.
    """
    # Ensure uniform input
    data_path = data_path if data_path[-1] == "/" else data_path + "/"

    if neural_path is None:
        neural_path = data_path + "neural/"
    neural_path = neural_path if neural_path[-1] == "/" else neural_path + "/"
    pickle_path = data_path + "pickles/"

    # Load the separate numpy pickles into lists.
    spectros = []
    time = []
    psg = []
    nns = []
    
    if nn_base_name == None:
        nn_base_name = "_trained_cnn.h5"
        
    # next(os.walk)[0] is the top-level directory. 
    # This is where the by-subject pickle folders should be.
    # next(os.walk)[1] is the list of files in next(os.walk)[0] 
    #   (not folders, which is next(os.walk)[2])
    list_of_subjects = next(os.walk(pickle_path))[1]
    if exclude is not None:
        list_of_subjects = [s for s in list_of_subjects if s not in exclude]

    for s in list_of_subjects:
        
        with open(pickle_path + s + "/spectros.pickle", "rb") as f:
            spectros.append(pickle.load(f))
        with open(pickle_path + s + "/time.pickle", "rb") as f:
            time.append(pickle.load(f))
        with open(pickle_path + s + "/psg.pickle", "rb") as f:
            psg.append(pickle.load(f))
        if nn:
            nns.append(tf.keras.models.load_model(neural_path + s + nn_base_name))

    for i in range(len(list_of_subjects)):
        if nn:
            yield list_of_subjects[i], spectros[i], time[i], psg[i], nns[i]
        else:
            yield list_of_subjects[i], spectros[i], time[i], psg[i]
        
def split_yielder(data_path = None, exclude = None, with_time = True):
    """
    Yields: (subject number excluded from training split, training_split_data, testing_split_data)
    Where:
    training_split_data:
      [0] if with_time, a 2-tuple:
        [0] training spectrograms tensor, shape (*, 65, 73)
        [1] training time tensor, shape (*,)
        (!) if not with_time, just training spectrograms
      [1] training psg labels tensor, shape (*,)
    testing_split_data is similar.

    See data_yielder for an explanation of the parameters.
     """
    data_list = [d for d in data_yielder(data_path = data_path, exclude = exclude, nn = False)]  
    
    for i, d in enumerate(data_list):
        if i == 0:
            skip_d = data_list[1:]
        else:
            skip_d = data_list[:i-1] + data_list[i:]
            
        train_spectro = np.concatenate([s[1] for s in skip_d])        
        train_psg = np.concatenate([s[3] for s in skip_d])
        
        
        if with_time:
            train_time = np.concatenate([s[2] for s in skip_d])
            training_split_data = ((train_spectro, train_time), train_psg)
            testing_split_data = ((d[1], d[2]), d[3])
        else:
            training_split_data = (train_spectro, train_psg)
            testing_split_data = (d[1], d[3])
    
        yield (d[0], training_split_data, testing_split_data)
      



def eval_nn(subject_number, nn_base_name, with_time = False, data_path = None, neural_path = None):
    """
    Helper function to load one subject's data and the neural net training on the complementary data, 
    then evaluate the neural network on the testing data. Returns a numpy array with shape
    (N, n_classes) where the subject's PSG has shape (N, 1) or (N, ).

    data_path and neural_path default to ../data/pickles and ../data/neural
    """
    # Set up paths
    if data_path is None:
        data_path = "data/pickles/"
    else:
        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string.")

        if data_path[-1] != "/":
            data_path += "/"

    if neural_path is None:
        neural_path = "data/neural/"
    else:
        if not isinstance(neural_path, str):
            raise ValueError("neural_path must be a string.")

        if neural_path[-1] != "/":
            neural_path += "/"

    # Load the neural network
    try:
        nn = tf.keras.models.load_model(neural_path + subject_number + nn_base_name)
    except:
        print("There was an error opening the trained neural network.")
        return

    # last layer of network has shape (None, n_classes)
    n_classes = nn.output.shape[1]

    # Load the subject's data, throwing exceptions if the pickles are missing or unopenable.
    ## Load spectrograms
    try:
        with open(data_path + subject_number + "/spectros.pickle", "rb") as f:
            spectros = pickle.load(f)
    except:
        print("There was an error opening the pickled spectrograms for {}.".format(subject_number))
        return
    ## Load labels
    try:
        with open(data_path + subject_number + "/psg.pickle", "rb") as f:
            psg = pickle.load(f)

        psg = sleep_classes(psg, n_classes)

    except:
        print("There was an error opening the pickled PSG for {}.".format(subject_number))
        return

    # Load time if we want that, and set up the testing data.
    if with_time is True:
        try:
            with open(data_path + subject_number + "/time.pickle", "rb") as f:
                time = pickle.load(f)
        except:
            print("There was an error opening the pickled times for {}.".format(subject_number))
            return

        testing_X = [spectros, time]

    else:
        testing_X = spectros

    

    return nn.predict(testing_X)

def pr_roc(subject_number, nn_base_name, pos_class = 0, with_time = False, data_path = None, neural_path = None, mode = "roc"):
    """
    Helper function to load one subject's data and the neural net training on the complementary data, 
    then evaluate the performance of this network via PR or ROC curve, depending on mode. 

    When pos_class is an int, returns a tuple suitable for plotting, interpolation, and pointwise averaging:
        (false positive rate, true positive rate) 
    OR
        (recall, precision)
    depending on if mode == "roc" or "pr", respectively.

    When pos_class is list<int>, returns a list with len(pos_class) elements. 
        The elements correspond to applying this function with the various pos_classes in the list.
        E.g. if pos_class = [0,1] then returns 
        [ pr_roc(..., pos_class = 0, ...), pr_roc(..., pos_class = 1, ...) ]

    data_path and neural_path default to ../data/pickles and ../data/neural
    """
    if not (isinstance(pos_class, int) or isinstance(pos_class, list)):
        raise ValueError("pos_class must be an int or list of ints")

    # Set up paths
    if data_path is None:
        data_path = "data/pickles/"
    else:
        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string.")

        if data_path[-1] != "/":
            data_path += "/"

    if neural_path is None:
        neural_path = "data/neural/"
    else:
        if not isinstance(neural_path, str):
            raise ValueError("neural_path must be a string.")

        if neural_path[-1] != "/":
            neural_path += "/"

    # Load the subject's data, throwing exceptions if the pickles are missing or unopenable.
    ## Load spectrograms
    try:
        with open(data_path + subject_number + "/spectros.pickle", "rb") as f:
            spectros = pickle.load(f)
    except:
        print("There was an error opening the pickled spectrograms for {}.".format(subject_number))
        return
    ## Load labels
    try:
        with open(data_path + subject_number + "/psg.pickle", "rb") as f:
            psg = pickle.load(f)
    except:
        print("There was an error opening the pickled PSG for {}.".format(subject_number))
        return

    # Load time if we want that, and set up the testing data.
    if with_time is True:
        try:
            with open(data_path + subject_number + "/time.pickle", "rb") as f:
                time = pickle.load(f)
        except:
            print("There was an error opening the pickled times for {}.".format(subject_number))
            return

        testing_X = [spectros, time]

    else:
        testing_X = spectros

    
    # Load the neural network
    try:
        nn = tf.keras.models.load_model(neural_path + subject_number + nn_base_name)
    except:
        print("There was an error opening the trained neural network.")

    outcome = nn.predict(testing_X)

    # Evaluate the outcomes, either operating on a single class if pos_class is an integer, or once per class given.
    eval_fn = roc_curve if mode == "roc" else precision_recall_curve
    if isinstance(pos_class, int):
        x, y, _ = eval_fn(y_true = psg, y_score = outcome[:, pos_class], pos_label = pos_class)
        return (x, y)

    else: # isinstance(pos_class, list)
        xys = [
            eval_fn(
                y_true = psg, 
                y_score = outcome[:, p], 
                pos_label = pos_class) 
            for p in pos_class
            ]

        if len(xys) == 1:
            xys = xys[0]

        return xys



def pointwise_avg(xy_pairs):
    """
    Given a list of (x, y) pairs, returns the pointwise average of
    the y-values viewed as functions of their x-values. 

    To accomplish this, we first take the refinement (union) of all the x's
    by what is essentially a merge sort + unique of these arrays. 
    """
    if len(xy_pairs) == 1:
        return xy_pairs

    xs = [np.array(x) for x, y in xy_pairs]
    
    refined = xs[0]
    for x in xs[1:]:
        refined = np.hstack([refined, x])
        refined.sort()
        refined = np.unique(refined)
    
    # Interpolate to all be evaluated on the full refinement
    ys = [np.interp(refined, x, y) for x, y in xy_pairs]

    return np.mean(ys, axis=0)
    

"""
TODO(ericcanton): use pr_roc function instead of repeating code here
Parameters:
    * data_yielder should be a generator or list that produces tuples of the form:
    ~~~ (subject name, spectrogram data, time data, PSG label data, neural network)
      Conveniently, the directory ../data/pickles/ has folders:
      ../data/pickles/#######/
      |------ spectros.pickle
      |------ time.pickle
      |------ psg.pickle
      |------ trained_cnn.h5
      so an os.walk can easily be turned into the desired yielder.
    * pos_label is the number of the class to be considered "positive". Defaults to 0.
    * label_names is an optional dictionary of names to use for labeling the 
    * saveto is a string with file path. The ROC plot will be saved as a PNG here. 
    ~~~ saveto = None (the default) saves nothing.
Returns:
    * (succeeded, evaluations, (fpr_interpolated, tpr_interpolated))
"""
def pr_roc_from_path(
        pickle_path,
        neural_path,
        nn_base_name,
        title=None, 
        pos_class = 0,
        n_classes=2, 
        label_names : dict = None, 
        saveto=None, 
        from_logits = True,
        with_time = False):

    import re

    if mode not in ['pr', 'roc']:
        raise ValueError("mode must be 'pr' OR 'roc'.")

    # Get list of NNs we have trained. These are the only ones we can evalutate!
    nns = next(os.walk(neural_path))[-1]
    subs = [re.search("[0-9]*", nn).group(0) for nn in nns]

    # Make all of the plots. 
    # We want to make an ROC and a PR plot for each class in pos_label.
    # First make sure we have a list
    if isinstance(pos_label, int):
        pos_label = [pos_label]

#    pr_roc_plots = {
#        'pr': [pr_roc(
#                subject_num = s, 
#                nn_base_name = nn_base_name, 
#                pos_class = pos_class,
#                with_time = with_time, 
#                data_path = pickle_path, 
#                neural_path = neural_path, 
#                mode = 'pr')
#            for s in subs],
#        'roc': [pr_roc(
#                subject_num = s, 
#                nn_base_name = nn_base_name, 
#                pos_class = pos_class,
#                with_time = with_time, 
#                data_path = pickle_path, 
#                neural_path = neural_path, 
#                mode = 'roc')
#            for s in subs]
#    }
    
    # Get list of per-class probability tensors
    outcomes = [eval_nn(
        subject_num = s,
        nn_base_name = nn_base_name,
        n_classes = n_classes,
        with_time = with_time,
        data_path = pickle_path,
        neural_path = neural_path)
        for s in subs]




    print("per-class (x,y) pairs created, where (x,y) = (FPR, TPR) or = (recall, precision).")

    fig, ax = plt.subplots(nrows=n_classes, ncols=2, figsize=(10*n_classes, 20))
    ax.grid(True)

    

    ax.margins(0.01)
    
    # Set up 'no-skill' baselines
    
    if mode == "roc":
        ax.plot([0, 1], '--', c='#000000')
        if label_names:
            xlabel = "%s error rate" % label_names["negative"]
            ylabel = "%s accuracy" % label_names["positive"]
        else:
            xlabel = "False positive rate"
            ylabel = "True positive rate"

    else: # precision-recall mode
        # A 'no-skill' predictor that guesses classes uniformly at random will have
        # - constant precision equal to the fraction of positive classes in the data.
        # - constant recall 0.5
        #p_pos = len(psg[psg == pos_label])/len(psg)
        #ax.plot([0,1], [p_pos, p_pos]) # 'no-skill' line
        
        if label_names:
            xlabel = "%s recall" % label_names["positive"]
            ylabel = "%s precision" % label_names["positive"]
        else:
            xlabel = "Recall"
            ylabel = "Precision"
        
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    if title:
        ax.set_title(title, fontsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    if saveto:
        plt.savefig(saveto)
    
    return [(s, auc(curve[0], curve[1])) for curve, s in zip(pr_rocs, succeeded)]




def get_probabilities(data_yielder, n_classes=2, from_logits=True):
   
    subjects = []
    evaluations = []
    psgs = []
    pr_rocs = []
    failed = []
    succeeded = []
    for subject, sp, time, psg, nn in data_yielder:
        print(subject)
        psg = sleep_classes(psg, n_classes)

        if from_logits:
            evaluations.append(logits_to_proba(nn.predict([sp, time])))
        else:
            evaluations.append(nn.predict([sp, time]))
        
        subjects.append(subject)
        psgs.append(psg)
    
    return subjects, psgs, evaluations

#
#"""
# Displays 3x3 grid of spectrograms, sampled randomly from a 
# numpy tensor, Xs, whose axis-0 cross-sections are the spectrograms. 
# Each spectrogram is labeled with the axis-0 index and class label from ys.
# 
# - Xs should be the spectrograms, ys the PSG labels. 
# -- Uses matplotlib.pyplot.pcolormesh, so this can be
#    used for displaying random axis-0 cross-sections of numpy 3-tensors.
# - ys should be the class labels
#"""
#def sample_pics(Xs, ys = None, with_maxmin = False):
#    import random
#    
#    Xs_len = Xs.shape[0]
#    if with_maxmin:
#        color_min = Xs.min()
#        color_max = Xs.max()
#    else:
#        color_min = None
#        color_max = None
#    
#    # Randomly select 9 indices for plotting the colormesh of their stft.
#    # Running this cell repeatedly gives a new subset each time.
#    inds = random.sample(range(Xs_len), k=9)
#    
#    # mapping dict taking pairs (i,j) to our random indices
#    # Used in for loop to plot random subset of our data.
#    mapper = {
#        (0,0) : inds[0],
#        (0,1) : inds[1],
#        (0,2) : inds[2],
#        (1,0) : inds[3],
#        (1,1) : inds[4],
#        (1,2) : inds[5],
#        (2,0) : inds[6],
#        (2,1) : inds[7],
#        (2,2) : inds[8],
#    }
#    
#    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(10,10))
#    
#    for i in [0,1,2]:
#        for j in [0,1,2]:
#            index = mapper[(i,j)]
#            X = Xs[index]
#            axs[i][j].pcolormesh(X, vmin=color_min, vmax=color_max)
#            title = "Index: {}".format(index)
#            if ys:
#                title += ", Class: {}".format(ys[index])
#                
#            axs[i][j].set_title(title)
#    plt.tight_layout()
