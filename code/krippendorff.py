import numpy as np
import pandas as pd
import krippendorff

df = pd.read_csv('Annotation_results.csv')
arr = df[['Piotr', 'Ondrej', 'Jakub', 'Hanka']].to_numpy().T
alpha = krippendorff.alpha(reliability_data=arr, level_of_measurement='ordinal')
print(alpha)
