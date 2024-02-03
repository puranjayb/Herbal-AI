from keras.models import load_model
import pandas as pd
import numpy as np
import cv2

model = load_model('Model_Weights/Herbal-AI-Classification.h5')
df = pd.read_csv('Model/index.csv')

IMAGE_SIZE = (160,160)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image, axis=0)
    image = np.array(image, dtype='float32')
    image = image / 255.0
    return image

def classify_image(image):
    prediction = model.predict(image)
    index = np.argmax(prediction[0])
    confidence = prediction[0][index]
    return index, confidence

def get_class(index):
    return df.loc[df['Index'] == index, 'Class'].values[0]

def classify_outlier(index, confidence_threshold=0.7):
    if confidence_threshold and confidence_threshold < 1.0:
        if confidence < confidence_threshold:
            return "Unknown or Outlier"
    return get_class(index)


test_image_path = 'Kakashi.jpg'
test_image = preprocess_image(test_image_path)
class_index, confidence = classify_image(test_image)
classification_result = classify_outlier(class_index)

print('*****')
print(type(classification_result))
print(classification_result)
print('*****')
