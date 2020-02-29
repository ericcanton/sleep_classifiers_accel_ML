# slcl_unsupervised
Learning unsupervised learning by analyzing sleep data from Olivia Walch's (@ojwalch) <a href="https://github.com/ojwalch/sleep_classifiers">sleep_classifiers</a>. 

My real goal is to familiarize myself with TensorFlow and/or PyTorch on a data set that I find personally interesting. Before doing any of that, I want to explore some of Scikit-Learn's unsupervised clustering algorithms, for a baseline agains which to compare neural network approaches.  

After an initial _k_-means clustering, with two clusters that should correspond to sleep/wake, I went on to try DBSCAN. The implementation is very expensive, both in terms of time and memory (easily filling 6+ GB of memory before dying). Thus, I switched to exploring the OPTICS algorithm. While Scikit-Learn's documentation estimates that OPTICS is O(n^2) (n=number of data points), it has the advantage of O(n) space complexity. With the ~300,000 data points in one subject's record from Walch's study, OPTICS finished in about 30 minutes running on one core of a 3.4GHz i3 processor. 
