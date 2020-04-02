# Machine Learning models based on acceleration from sleep_classifiers
Learning supervised and unsupervised machine learning by analyzing sleep data from Olivia Walch's (@ojwalch) <a href="https://github.com/ojwalch/sleep_classifiers">sleep_classifiers</a>. 

This is my way to grow my experience building and applying machine learning models in Python by analyzing timeseries from the aforementioned <code>sleep_classifiers</code>. Guided by Olivia's suggestions...
1. I first evaluated the practicality of using K-means and OPTICS clustering algorithms for unsupervised learning; these attempts you can find in the <code>unsupervised.ipynb</code> Jupyter Notebook. Neither was very effective at predicting if a subject from the study was asleep or awake. 
2. More recently,Olivia and I have been exploring the use of short-term Fourier transforms (STFTs) of the derivative of accelerometer data. In <code>log_regression.ipynb</code>, I flesh out an idea Olivia had to use supervised ML on averaged STFTs of the 30-second intervals where PSG labeled subjects as awake or asleep. To give some idea for the plausibility of this, below you will see a spectrogram of this accelerometer differential along with a scatterplot of the differential (in blue) and the PSG labels (in maize).   
<img src="Images/8173033_spectrogram_PSG.png">  
I used a logistic regression model, and ended up with ~94% precision, ~98% recall. Future directions include expanding the sleep stages upon which the logistic model is trained. 
