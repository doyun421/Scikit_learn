
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

from sklearn import linear_model

# Load CSV and columns
df_X = pd.read_csv("./ticktick/data/metadata.csv")
df_Y = pd.read_csv("./ticktick/data/train_labels.csv")

X_longitude = []
X_latitude = []
X_date = []
X_uid = []

for i in range(0, len(df_X['split'])):
    if df_X['split'][i] == "train":

            X_uid.append(df_X['uid'][i])

            X_latitude.append(df_X['latitude'][i])
            X_longitude.append(df_X['longitude'][i])
             
            X_date.append(df_X['date'][i])

Y_uid = df_Y['uid']
Y_region = df_Y['region']
Y_severity = df_Y['severity']
Y_density = df_Y['density']

reg = linear_model.LinearRegression()
print(len(X_date), len(df_Y['severity']))
X_latitude = np.reshape(X_latitude, (-1, 1))
print(X_latitude)

Y_severity = np.reshape(Y_severity, (-1))
print(Y_severity)





reg.fit(X_latitude, Y_severity)

print(reg.coef_)


# Plot outputs
plt.scatter(X_latitude, Y_severity,  color='black')
plt.title('Train Data')
plt.xlabel('latitude')
plt.ylabel('severity')
plt.xticks(())
plt.yticks(())

plt.show()
