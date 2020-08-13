""" 
Evaluation scripts. There are four functions defined here:
    1. logits_to_proba @ line 34
    2. data_yielder @ line 42
    3. split_yielder @ line  92
    4. pr_roc_from_path @ line 145
    5. sample_pics @ line 279
    6. get_probabilities @ line 270
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import tensorflow as tf

from sklearn.metrics import roc_curve, precision_recall_curve, auc

import os
import pickle

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
def logits_to_proba(logit):
    return np.exp(logit)/(1+np.exp(logit))

def sleep_classes(inputs, n_classes):
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


# Generator to create leave-one-out train/test splits.
# Upon initialization, loads the separate data pickles.
# Yields: (subject_number[i], spectrogram[i], time[i], psg labels[i], neural network[i])
## If exclude != None, then it's a list of subject numbers not to be yielded.
def data_yielder(data_path, neural_path = None, exclude = None, nn = True, nn_base_name = None):
    # Ensure uniform input
    data_path = data_path if data_path[-1] == "/" else data_path + "/"

    if neural_path == None:
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
        
# See data_yielder for an explanation of the parameters.
# Yields: (subject number excluded from training split, training_split_data, testing_split_data)
## Where:
# training_split_data:
#   [0] if with_time, a 2-tuple:
#     [0] training spectrograms tensor, shape (*, 65, 73)
#     [1] training time tensor, shape (*,)
#     (!) if not with_time, just element [0]
#   [1] training psg labels tensor, shape (*,)
# testing_split_data is similar.
def split_yielder(data_path = None, exclude = None, with_time = True):
            
    """
    Each d in ev.data_yielder(...) is a 4-tuple:
      [0] this subject's number
      [1] this subject's spectrogram tensor, shape (*, 65, 73)
      [2] this subject's time tensor, shape (*,)
      [3] this subject's psg labels tensor, shape (*,)
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
def pr_roc_from_path(data_yielder, title=None, pos_label=0, n_classes=2, label_names : dict=None, saveto=None, axis=None, mode="roc", from_logits = True):
    if mode not in ['pr', 'roc']:
        raise ValueError("mode must be in ['pr', 'roc'].")

    #######################################################################################
    #######################################################################################
    # Evaluate the left-one-out test samples on the trained neural networks
    evaluations = []
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
    #    print("Evaluated...")

        # roc_curve(...) ~~> (FPR, TPR, thresholds)
        # evaluations[-1] is a numpy array with shape (*, 2)
        #   ~~~ evaluations[-1][i] == (Pr(y[i] = 0), Pr(y[i] = 1))
        #   Thus, we use evaluations[-1][:, pos_label] to get a vector of probabilites of y[i] == pos_label
        try:
            if mode == "roc":
                pr_rocs.append(roc_curve(y_true = psg, y_score=evaluations[-1][:,pos_label], pos_label = pos_label))
            else: 
                pr_rocs.append(precision_recall_curve(psg, evaluations[-1][:, pos_label], pos_label=pos_label))

            succeeded.append(subject)
        except:
            failed.append(subject)
            continue

    if len(failed) > 0:
        print("There were class issues for:")
        print(failed)
    if len(succeeded) == 0:
        return evaluations

    #######################################################################################
    #######################################################################################
    # Adjust the ROC outcomes so they're all the same length. That way, we can average them
    # each roc in pr_rocs has the form
    # roc == (FPR, TPR, thresholds)
    max_thresh_len = max([len(roc[2]) for roc in pr_rocs]) # max length of fpr vector
    fpr_interpolated = np.linspace(0, 1, max_thresh_len) # thresholds vector, for interpolation

    ## rocs_np[split #, threshold[j], fpr 0 / tpr 1]
    tpr_interpolated = np.zeros((len(pr_rocs), max_thresh_len))
    for j in range(len(pr_rocs)):
        ## pr_rocs[j] == (fpr, tpr, thresholds)
        tpr_interpolated[j] = np.interp(x =fpr_interpolated, xp=pr_rocs[j][0], fp=pr_rocs[j][1]) 

    roc_averaged = np.mean(tpr_interpolated, axis=0, dtype=np.float64)

    if axis == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    else:
        ax = axis

    ax.plot(fpr_interpolated, roc_averaged, c='#449deb', lw=2)
    for j in range(len(pr_rocs)):
        ax.plot(pr_rocs[j][0], pr_rocs[j][1], alpha=0.3, c='orange')

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
