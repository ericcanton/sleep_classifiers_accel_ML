import numpy as np
import pandas as pd

#################################################################
# Helper functions for magnitude of 3d data, numerical derivative
#################################################################

def mag(xyz : pd.DataFrame):
    return (xyz['x']**2 + xyz['y']**2 + xyz['z']**2)**(0.5)

def vder(f : pd.DataFrame):
    f_norm = f.reset_index(drop=True)
    t0, t1 = f_norm.iloc[:-1, 0], f_norm.iloc[1:, 0]
    f0, f1 = f_norm.iloc[:-1, 1], f_norm.iloc[1:, 1]

    # t1 and f1 are indexed 1..n
    # change to match t0, f0 indexed 0..(n-1)
    t1.reset_index(drop=True, inplace=True)
    f1.reset_index(drop=True, inplace=True)

    # Calculate the derivative
    df = (f1 - f0)/(t1 - t0)
    
    # Calculate basic time statistics
    w = (t1 - t0).describe()

    return (w, t0.to_numpy(), df.to_numpy())


def get_accel(subject : str, non_neg = True, path = "data/motion/"):
    # Import the acceleration data
    accel = pd.read_csv(path + "{}_acceleration.txt".format(subject), \
                        sep=' ', names=['time', 'x', 'y', 'z'])
    if non_neg:
        accel = accel[accel['time'] >= 0]
    accel.reset_index(drop=True, inplace=True)
    
    # Calculate magnitude of the (x,y,z) accelerations
    accel['mag'] = mag(accel[['x', 'y', 'z']])
    # then calculate numerical derivative of the acceleration magnitudes.
    tstat, t0, dmag = vder(accel[['time', 'mag']])

    dmag_df = pd.DataFrame({'time' : t0, 'dmag' : dmag})

    return (accel, dmag_df)

