import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()
train.shape
test.shape

import matplotlib.pyplot as plt
image = np.array(train.iloc[3,1:], dtype='float')
image = image.reshape((28,28))
plt.imshow(image, cmap='gray')
