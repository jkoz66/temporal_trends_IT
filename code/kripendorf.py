import numpy as np
import pandas as pd
import krippendorff

df = pd.read_csv('Annotations_wild_data - project-2-at-2025-12-05-13-15-7b69df03.csv.csv')
arr = df[['Piotr', 'Ondrej', 'Jakub', 'Hanka']].to_numpy().T
alpha = krippendorff.alpha(reliability_data=arr, level_of_measurement='ordinal')
print(alpha)