import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()
train.shape
test.shape

import matplotlib.pyplot as plt
# Selecting 4th row/number and all columns except Label
image = np.array(train.iloc[3,1:], dtype='float')
image = image.reshape((28,28))
plt.imshow(image, cmap='gray')

from sklearn.model_selection import train_test_split
X = train.iloc[:,1:]
y = train.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print('Accuracy Score : ', accuracy_score(y_train, dtc.predict(X_train)))
print('Accuracy Score : ', accuracy_score(y_test, dtc.predict(X_test)))
