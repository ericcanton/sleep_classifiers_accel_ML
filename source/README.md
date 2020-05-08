## ETL and Feature Extraction
Assuming you have already stored the accelerometer and PSG labeling data in <code>../data/motion/</code> and <code>../data/labels/</code>, resp., we now cleanse the data slightly while doing some feature extraction (generating spectrograms of derivatives of windowed accelerometer magnitudes) and then save it all to <code>../data/spectros/</code>. If you load <code>etl.py</code> as a script, three files, all pickled numpy arrays, are created by default. If you call <code>$ python etl.py --split-subs</code> then six are created: one train and one test file for each of the three files below.
1. <code>labels.pickle</code> contains timestamped class labels.
2. <code>spectro.pickle</code> contains the 3-tensor of spectrograms (stacked along axis 0).
3. <code>dmags.pickle</code> contains the windows (stacked along axis 0) whose spectrograms are taken.
