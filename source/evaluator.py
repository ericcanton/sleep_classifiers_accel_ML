import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import roc_curve, precision_recall_curve, auc

"""
Input: 
-- y_hat, list of predicted classes for sample sets
-- y_true, list with same len as y_hats giving true class
-- pos_label, an int specifying which class counts as "positive" for ROC
    Defaults to 0 "Awake"

Output:
-- areas, list of same len as y_hats containing ROC area under curve (AUC)
-- MatPlotLib Axes object, the ROC plot
-- ROC plot saved to disk
"""
def roc_one_class(y_hats, y_truths, title = None, pos_label = 0, saveto=None, axis = None):
    if not axis:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    else:
        ax = axis
    ax.plot([0, 1], '--', c='#854208')
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # returned list of AUC scores
    areas = np.zeros(len(y_hats))

    count = 0
    for hat, true in zip(y_hats, y_truths):
        fpr, tpr, _ = roc_curve(y_true = true, y_score = hat, pos_label = pos_label)
        areas[count] = auc(fpr, tpr)
        ax.plot(fpr, tpr, label="AUC: {:5.4f}".format(areas[count]))

        count += 1

    ax.legend(loc='lower right')
    
    ax.grid(True)
    
    ax.margins(0.05) 

    if saveto:
        plt.savefig(saveto)

    return areas


def roc_multi_class(y_hats, y_truths, classes = None, class_names = None, title = None, saveto=None):
    # Default behavoir when no list of classes is provided.
    # y_hats[0] should be a 2D numpy array. Rows (axis 1) give class probabilities
    # Figure out number of classes by the "width" of this array.
    if classes == None:
        classes = [i for i in range(y_hats[0].shape[1])]

    fig, axs = plt.subplots(nrows = len(classes), ncols = 1, figsize=(20, 5+10*len(classes)), tight_layout = True)
    for c in classes:
        if class_names:
            print("Working on class %s" % class_names[c])
            this_title = "%s" % class_names[c]
        else:
            print("Working on class %d" % c)
            this_title = "Class %d" % c

        these_hats = [h[:, c] for h in y_hats]
        roc_one_class(these_hats, y_truths, pos_label=c, title=this_title, axis=axs[c])

    if title:
        plt.suptitle(title, fontsize=20)
    if saveto:
        plt.savefig(saveto)

    return (fig, axs)

"""
data should be a list of lists with structure:
    -- data[0] = training data
    -- data[1] = testing data
    each data[i] has two elements:
    -- data[i][0] = model inputs. Either spectrogram tensor, or [spectrogram, times] 
    -- data[i][1] = PSG labels tensor. DO NOT PREPROCESS THIS INTO BI/TRINARY CLASSES 

nn_model should be a compiled Keras model, either trained or untrained. 
For each d in data_splits, a ("deep") copy will be made and trained for 15 epochs. 
"""
def train_then_ROC_PR(data, nn_model, uses_time = True, saveto_path = None, pos_label = 0):
    if not (isinstance(pos_label, int) or isinstance(pos_label, list)):
        raise ValueError("pos_label needs to be an integer or a list of length 3 or 4.")
        return None

    fprs = []
    tprs = []
    precs = []
    recas = []
    aucs = []

    training, testing  = data

    #####################################################################
    # Discard -1 ("unscored") labels, and corresponding data
    #####################################################################
    training_scored, testing_scored = (training[1] != -1), (testing[1] != -1)

    # Do PSG labels first
    training[1] = training[1][training_scored]
    testing[1] = testing[1][testing_scored]

    # Now reduce the input data accordingly
    if uses_time:
        # spectrograms
        training[0][0] = training[0][0][training_scored]
        testing[0][0] = testing[0][0][testing_scored]

        # time
        training[0][1] = training[0][1][training_scored]
        testing[0][1] = testing[0][1][testing_scored]
    else:
        training[0] = training[0][training_scored]
        testing[0] = testing[0][testing_scored]

    #####################################################################
    # Now modify the labels to give wake/sleep or staging 
    #####################################################################

    # Decide which label mode the user wants based on the type, and possibly length, of pos_label. 
    if isinstance(pos_label, int): # binary classifier (Awake/Asleep), default
        training[1] = np.where(training[1] > 0, 1, 0)
        testing[1] = np.where(testing[1] > 0, 1, 0)
        mode = 2

    elif isinstance(pos_label, list):
        if len(pos_label) == 3: # Awake/NREM/REM
            training[1] = np.piecewise( \
                training[1], \
                [ \
                 training[1] == 0, \
                 (0 < training[1]) & (training[1] < 5), \
                 training[1] == 5 \
                ], \
                [0,1,2] \
            )
            testing[1] = np.piecewise( \
                testing[1], \
                [ \
                 testing[1] == 0, \
                 (0 < testing[1]) & (testing[1] < 5), \
                 testing[1] == 5 \
                ], \
                [0,1,2] \
            )
            mode = 3
        else: # Awake/NREM12/NREM34/REM
            training[1] = np.piecewise(
                training[1], \
                [training[1] == 0, \
                 (0 < training[1]) & (training[1] < 3), \
                 (2 < training[1]) & (training[1] < 5), \
                 training[1] == 5
                ], \
                [0, 1, 2, 3] \
            )
            testing[1] = np.piecewise(
                testing[1], \
                [testing[1] == 0, \
                 (0 < testing[1]) & (testing[1] < 3), \
                 (2 < testing[1]) & (testing[1] < 5), \
                 testing[1] == 5
                ], \
                [0, 1, 2, 3] \
            )
            mode = 4

    print("Split okay! Using %d labels." % mode)
    print(training[1])
    print(testing[1])
    return (fprs, tprs, aucs, precs, recas)

    this_nn = deepcopy(nn_model)

    this_nn.fit(training[0], testing, epochs=15, verbose = 0)

    predictions = this_nn.predict(testing)

if __name__ == "__main__":
    training = [np.arange(0, 10, 1), np.random.randint(-1, 6, 10)]
    testing = [np.arange(0, 5, 1), np.random.randint(-1, 6, 5)]
    
    print(training[1])
    print(testing[1])

    train_then_ROC_PR([training, testing], "abc", uses_time=False, pos_label=[0,1,2])
