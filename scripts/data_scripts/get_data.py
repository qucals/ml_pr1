#!/usr/bin/python3

from catboost.datasets import titanic

train, test = titanic()

train.to_csv("/home/data-srv/ml_pr1/data/raw/train.csv", columns=train.columns)

test.to_csv("/home/data-srv/ml_pr1/data/raw/test.csv", columns=test.columns)