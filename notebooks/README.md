The first Jupyter Notebook, intended to be run on Google Colab, 

1. <code>stft_no_avg.ipynb</code> expects to be run on Google Colab, but you can find a link to the hosted Notebook in the first paragraph of this notebook. This notebook imports a pickled numpy 3-tensor whose axis-0 cross-sections are spectrograms of ~2.56 second windows of accelerometer derivatives; links to these data files on my Google Drive are included in the Notebook. 

I build two neural networks to classify the STFTs, which are interpreted more-or-less as black-and-white images. I also explore the change in classification capability when the NumPy arrays representing the STFTs have been normalized to have values in the interval [0, 255]; the normalization interval is chosen based on the usual intensity values for black-and-white images. 

For both networks, the unscaled data provides superior precision and recall for the Awake class compared to NNs trained on scaled data. The NNs trained on unscaled data also have superior performance to the logistic regression models. I have also included a logistic regression model at the end of this Notebook for comparison with the neural networks. The classes are weighted when fitting the sigmoid, since Awake is quite rare in the data. 

Below are the precision and recall for the Awake class for the DNN, CNN, and log regression models on a recent run. All three do fairly well for Asleep, so we do not include metrics here (you can find them in the Notebook). 

Model | Precision | Recall
| --- | --- | --- |
DNN, unscaled | 0.97 | 0.12
DNN, scaled | 0.75 | 0.02
CNN, unscaled | 0.71 | 0.29
CNN, scaled | 0.60 | 0.02
Log, unscaled | 0.21 | 0.52
Log, scaled | 0.32 | 0.15


2. In <code>log_regression.ipynb</code>, I flesh out an idea Olivia had to use supervised ML temporally-averaged STFTs. I used an unblanced logistic regression model, and ended up with ~94% precision and ~98% recall for Asleep, but the model mostly just predicts Asleep and performs poorly on Awake.

3. In <code>unsupervised.ipynb</code>, I evaluate the practicality of using _k_-means and OPTICS clustering algorithms for unsupervised learning directly on the (x, y, z)-acceleration data. Neither was very effective at predicting if a subject from the study was asleep or awake, and I haven't pusued this further. 
