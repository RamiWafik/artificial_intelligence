# Data EDA

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Importing dataset and cleaning data from NaN
dataset = pd.read_csv('financial_data.csv')
dataset.head()
dataset.columns
dataset.describe()
dataset.isna().any()

# Plotting Histograms
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Correlation with dependant variable
dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15, rot = 45, grid = True)

# Correlation between independant variables
corr = dataset.corr()
plt.subplots(figsize=(18, 15))
sns.heatmap(corr, annot = True)
