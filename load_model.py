import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import model_from_json
import numpy as np


# for i in range(4):
#     model_count = str(i)
#     if model_count == '0':
#         model_count = ''
#     with open(f'model{model_count}.json', 'r') as f:
#         model = model_from_json(f.read())
#     model.load_weights(f'model_weights{model_count}.weights.h5')
#
#
#     results = (model.predict(np.array([2.5])),
#                model.predict(np.array([7.95])),
#                model.predict(np.array([11.3])),
#                model.predict(np.array([-60.2])),
#                model.predict(np.array([100])),
#                model.predict(np.array([1000])),
#                model.predict(np.array([-763.5])),
#                model.predict(np.array([666])),
#                model.predict(np.array([6666])),
#                model.predict(np.array([66666])))
#
#
#     if model_count == '':
#         print(f'Модель 1: \n\n\n')
#     else:
#         print(f'Модель {i}: \n\n\n')
#     for result in results:
#         print(format(result[0][0], '.2f'))
#     print('\n\n\n')


with open(f'model1.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights(f'model_weights1.weights.h5')
results = (model.predict(np.array([114])),
           model.predict(np.array([-100])),
           model.predict(np.array([-235])),
           model.predict(np.array([-60.2])),
           model.predict(np.array([1234])),
           model.predict(np.array([12345])),
           model.predict(np.array([-763.5])),
           model.predict(np.array([7845])),
           model.predict(np.array([7777])),
           model.predict(np.array([777])))
for result in results:
    print(format(result[0][0], '.2f'))