# Directing customers to subscription

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

# Importing the dataset and cleaning
dataset = pd.read_csv('appdata10.csv')
dataset.head()
dataset.describe()
dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)
dataset.hour

# Data visualisation for numerical columns
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
dataset2.head()
plt.suptitle('Histograms of Numerical Columns', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = 'green')
    
# Correlation plot
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20, 10), 
                  title = 'Correlation With Response Variable',
                  fontsize = 15, rot = 45, grid = True)

# Confusion Matrix
corr = dataset2.corr()
sns.heatmap(corr, annot = True)

# Feature Engineering for response
dataset.dtypes
dataset['first_open'] = [parser.parse(row_date) for row_date in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset['enrolled_date']]
dataset.dtypes

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')
plt.hist(dataset['difference'].dropna())
plt.title('Distribution of time since enrolled')
plt.show()

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')
plt.hist(dataset['difference'].dropna(), range = [0, 100])
plt.title('Distribution of time since enrolled')
plt.show()

# Will take only first 48 hours - 2 days
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])

# Feature Engineering for screens

# Load Top Screens
top_screens = pd.read_csv('top_screens.csv').top_screens.values
top_screens
dataset["screen_list"] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "")

dataset['Other'] = dataset.screen_list.str.count(",")
dataset = dataset.drop(columns=['screen_list'])

# Funnels
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

## Saving New Dataset
dataset.head()
dataset.describe()
dataset.columns
dataset.hour

dataset.to_csv('new_appdata1.csv', index = False)
