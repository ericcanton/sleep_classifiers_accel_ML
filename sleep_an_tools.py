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