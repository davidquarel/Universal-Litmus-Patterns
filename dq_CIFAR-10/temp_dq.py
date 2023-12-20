# %%
import csv
import pandas as pd
data = []
path = "results/model_poisonacc_matched.csv"
with open(path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(float(row['accuracy']))
# %%