import sys
import os
import pickle
import yaml

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 dt.py data-file.csv model\n")
    sys.exit(-1)
    
f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

params = yaml.safe_load(open("settings/params.yaml"))["train"]
random_state = params["random_state"]
max_depth = params["max_depth"]

df = pd.read_csv(sys.argv[1], header=None)
X_train = df.iloc[:, [0, 1, 2]]
y_train = df.iloc[:, 3]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.transform(y_train)

model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

with open(f_output, "wb") as fd:
    pickle.dump(model, fd)