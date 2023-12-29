import sys
import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 dt.py data-file.csv model\n")
    sys.exit(-1)
    
f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])

os.makedirs(os.path.join("models"), exist_ok=True)

# TODO: Continue to write