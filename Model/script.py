from keras.models import load_model
import pandas as pd
import numpy as np
import cv2

model = load_model('Model_Weights/Herbal-AI-Classification.h5')
df = pd.read_csv('Model/index.csv')

IMAGE_SIZE = (160,160)

image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, IMAGE_SIZE)
image = np.expand_dims(image, axis=0)
image = np.array(image, dtype = 'float32')
image = image/255.0

prediction = model.predict(image)
index = np.argmax(prediction[0])
classification_result = df.loc[df['Index'] == index, 'Class'].values[0]

print('*****')
print(type(classification_result))
print(classification_result)
print('*****')

