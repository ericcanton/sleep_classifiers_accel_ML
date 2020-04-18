# Machine Learning models based on acceleration from sleep_classifiers
Learning supervised and unsupervised machine learning by analyzing sleep data from Olivia Walch's (@ojwalch) <a href="https://github.com/ojwalch/sleep_classifiers">sleep_classifiers</a>. 

In this project, I analyze timeseries from the aforementioned <code>sleep_classifiers</code> with the goal of predicting sleep stages (awake/NREM/REM) based on Apple Watch accelerometer data. The current focus is accurate prediction of Awake/Asleep. For supervised ML, I build two TensorFlow neural networks and two logistic regression models. The first NN has only densely connected layers, the second has two convolutional layers, separated by a max pool layer, that feed into two densely connected layers. Additionally, I also explore _k_-means and OPTICS for unsupervised clustering.  

Dr. Walch and I have been exploring the use of short-term Fourier transforms (STFTs) of the derivative of accelerometer data for supervised ML. The inspiration is the use of this technique for voice recognition software, which provides the ability to identify not only words but speaker based on characteristics of the frequencies in a given small time window. When plotted with time on the horizontal axis and frequency on the vertical axis, we can visualize these STFTs as spectrograms. These spectrograms are input as black-and-white images to the neural networks. To give some idea for the information contained in these, here is a spectrogram (top) generated using the Librosa library along with the derivative of acceleration (in blue)and PSG labels (in maize) on the bottom. 
<img src="Images/8173033_spectrogram_PSG.png">  

The first Jupyter Notebook, intended to be run on Google Colab, imports a file of <code>pickled</code> STFTs; links to these data files on my Google Drive are included in the Notebook. 
The second and third Jupyter Notebooks assume you have already extracted the data from Dr. Walch's study into the <code>data</code> folder of the directory in which you're running them.

1. <code>stft_no_avg.ipynb</code> expects to be run on Google Colab, but you can find a link to the hosted Notebook in the first paragraph of this notebook. 

Here, I build two neural networks to classify the STFTs, which are interpreted more-or-less as black-and-white images. I also explore the change in classification capability when the NumPy arrays representing the STFTs have been normalized to have values in the interval [0, 255]; the normalization interval is chosen based on the usual intensity values for black-and-white images. 

For both networks, the unscaled data provides superior precision and recall for the Awake class compared to NNs trained on scaled data. The NNs trained on unscaled data also have superior performance to the logistic regression models. I have also included a logistic regression model at the end of this Notebook for comparison with the neural networks. The classes are weighted when fitting the sigmoid, since Awake is quite rare in the data. 

Below are the precision and recall for the Awake class for the DNN, CNN, and log regression models on a recent run. All three do fairly well for Asleep, so we do not include metrics here (you can find them in the Notebook). 

                Precision   Recall

DNN, unscaled   0.97        0.12

DNN, scaled     0.75        0.02

CNN, unscaled   0.71        0.29

CNN, scaled     0.60        0.02

Log, unscaled   0.21        0.52

Log, scaled     0.32        0.15


2. In <code>log_regression.ipynb</code>, I flesh out an idea Olivia had to use supervised ML temporally-averaged STFTs. I used an unblanced logistic regression model, and ended up with ~94% precision and ~98% recall for Asleep, but the model mostly just predicts Asleep and performs poorly on Awake.

3. In <code>unsupervised.ipynb</code>, I evaluate the practicality of using _k_-means and OPTICS clustering algorithms for unsupervised learning directly on the (x, y, z)-acceleration data. Neither was very effective at predicting if a subject from the study was asleep or awake, and I haven't pusued this further. 
