The latest iteration of our work, including our best-performing neural network definitions and leave-one-out training/evaluation, are found in <code>Sharable_leave_one_out.ipynb</code>. That notebook contains more discussion of our methods at the top; feel free to run this on your own machine, or make a copy of the notebook on Google Colab (link in the notebook). Training a convolutional neural network without GPU access is a time-intensive process, but Google gives access to their GPUs on Colab.

For prior approaches, look in the <code>old</code> directory.  
- You can find simplified neural network architectures trained and evaluated on 80/20 train/test splits in <code>old/stft_no_avg.ipynb</code>. Therein you will also find a logistic regression model.  
- <code>old/log_regression.ipynb</code> fleshes out an idea Olivia had for sleep/wake prediction using log regression on temporally-averaged spectrograms; the model mostly just predicts "Asleep". 
- <code>old/unsupervised.ipynb</code> contain some exploratory evaluation of , _k_-means and OPTICS for unsupervised clustering. 
