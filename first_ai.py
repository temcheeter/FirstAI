import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


input_nums = ['count']
output_nums = [' square_of_count']
data_frame = pd.read_csv('data.csv', sep=';')

train_x = np.array(data_frame[input_nums])
train_y = np.array(data_frame[output_nums])


model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1, input_dim=1, activation='relu'))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
x = np.array([i for i in range(1, 1001)])
y = np.array([i * i for i in range(1, 1001)])


fit_results = model.fit(x, y, epochs=100, validation_split=0.1)


# xe = np.array([109, 115, -178, 235, 60.2, -100, 1000, -763.5, 666, 6666, 66666])
# ye = np.array([11881, 13225, 31684, 55225, 3624.04, 10000, 1000000, 582932.25, 443556, 44435556, 4444355556])


print(model.predict(np.array([777])))

# saving model
model_json = model.to_json()
with open('model1.json', 'w') as f:
    f.write(model_json)
model.save_weights('model_weights1.weights.h5')
print('model saved')


plt.title('Losses train')
# plt.plot(fit_results.history['loss'], label='Train')
# plt.plot(fit_results.history['val_loss'], label='Validation')
plt.plot(10, label='x')
plt.plot(10 ** 2, label='x^2')
plt.legend()
plt.show()