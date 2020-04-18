# Machine Learning models based on acceleration from sleep_classifiers
Learning supervised and unsupervised machine learning by analyzing sleep data from Olivia Walch's (@ojwalch) <a href="https://github.com/ojwalch/sleep_classifiers">sleep_classifiers</a>. 

In this project, I analyze timeseries from the aforementioned <code>sleep_classifiers</code> with the goal of predicting sleep stages (awake/NREM/REM) based on Apple Watch accelerometer data. As a first step, we aim to accurately decide Awake/Asleep. For supervised ML, I build a logistic regression model and two TensorFlow neural networks. The first NN has only densely connected layers, the second has two convolutional layers, separated by a max pool layer, that feed into two densely connected layers. I also explore _k_-means and OPTICS for unsupervised clustering.  

Olivia and I have been exploring the use of short-term Fourier transforms (STFTs) of the derivative of accelerometer data. The inspiration is the use of this technique for voice recognition software, which provides the ability to identify not only words but speaker based on characteristics of the frequencies in a given small time window. When plotted with time on the horizontal axis and frequency on the vertical axis, we can visualize these STFTs as spectrograms. These spectrograms are input as black-and-white images to the neural networks. To give some idea for the information contained in these, here is a spectrogram (top) along with the derivative of acceleration (in blue)and PSG labels (in maize). 
<img src="Images/8173033_spectrogram_PSG.png">  

These projects assume you have already extracted the data from Dr. Walch's study into the <code>data</code> folder of the directory in which you're running this code. 
The Colab Notebook also imports a file of <code>pickled</code> STFTs; links to these data files are included in the Notebook. 

Guided by Dr. Walch's suggestions, the machine learning components are broken into the following Notebooks. 
1. <code>stft_no_avg.ipynb</code> expects to be run on Google Colab, but you can find a link to the hosted Notebook in the first paragraph of this notebook. 

Here, I build two neural networks (as described above) to classify the STFTs, interpreted as black-and-white images. I also explore the change in classification capability when the NumPy arrays representing the STFTs have been normalized to have values in the interval [0, 255]; the normalization interval is chosen based on the usual intensity values for black-and-white images. 

For both networks, the un-scaled data provides superior precision and recall for the Awake class. Numerical variability of these metrics is high, and sadly the CNN trained on scaled data often returns no instances of Awake, thus giving precision/recall of 0.0. I have also included a logistic regression model at the end of this Notebook for comparison with the neural networks. The classes are weighted when fitting the sigmoid, since Awake is quite rare in the data. 

2. In <code>log_regression.ipynb</code>, I flesh out an idea Olivia had to use supervised ML on averaged STFTs of the 30-second intervals where PSG labeled subjects as awake or asleep. I used an unblanced logistic regression model, and ended up with ~94% precision and ~98% recall for Asleep, but the model mostly just predicts Asleep and performs poorly on Awake.

3. In <code>unsupervised.ipynb</code>, I evaluate the practicality of using _k_-means and OPTICS clustering algorithms for unsupervised learning directly on the (x, y, z)-acceleration data. Neither was very effective at predicting if a subject from the study was asleep or awake, and I haven't pusued this further. 
