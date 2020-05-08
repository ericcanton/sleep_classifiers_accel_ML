import numpy as np

from scipy.signal import lombscargle as lssa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Compute Lomb-Scargle periodogram in 30 second chunks, with 10 second overlap.
# First time period is 20 seconds long.
def lssa_chunks(time : np.ndarray, y : np.ndarray, overlap = 10, step = 20, max_freq = 4*np.pi/0.5): 
    if step <= 0:
        raise ValueError("step must be positive.")

    fr = np.linspace(1e-4, max_freq + 1e-4, 500)

    t_min = time.min()
    t_max = time.max()

    masks = [(t_min <= time) & (time <= t_min + step)] 
    masks += [(t_min + n*step - overlap <= time) & (time <= t_min + (n+1)*step) 
              for n in range(1, int((t_max - t_min)/step)-1)]

    times = [time[mask] for mask in masks]
    ys = [y[mask] for mask in masks]

    lssas = [lssa(t, y, fr) for t, y in zip(times, ys)]

    return (fr, lssas)

def log_model(X : np.ndarray, y : np.ndarray):
    sgd = SGDClassifier(loss='log')

    sgd.fit(X, y)

    return sgd

