import os
import shutil
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array, array_to_img
from sklearn.utils import shuffle

from keras import layers
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler

# ------------------------------------------------------------------------------------------------------------------------------------------------------

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
        )
else:
    print("No GPUs available.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------

input_data_dir = '../Dataset/Plants'
output_data_dir = '../Dataset/Plants'
target_num_images = 250
def augment_and_save_images(class_folder, image_names):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    output_class_path = os.path.join(output_data_dir, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    num_images_to_generate = target_num_images - len(image_names)

    if num_images_to_generate > 0:
        for i in tqdm(range(num_images_to_generate), desc=f'Augmenting {class_folder}'):
            random_image_name = random.choice(image_names)
            image_path = os.path.join(input_data_dir, class_folder, random_image_name)

            img = load_img(image_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for batch in datagen.flow(x, batch_size=1):
                augmented_img = array_to_img(batch[0])
                augmented_image_name = f"augmented_{random.randint(1, 100000)}.jpg"
                output_path = os.path.join(output_class_path, augmented_image_name)
                augmented_img.save(output_path)
                break


for class_folder in os.listdir(input_data_dir):
    class_path = os.path.join(input_data_dir, class_folder)
    existing_images = [image_name for image_name in os.listdir(class_path) if image_name.endswith('.jpg')]
    augment_and_save_images(class_folder, existing_images)


# ------------------------------------------------------------------------------------------------------------------------------------------------------

root_directory = '../Dataset/Plants'

class_names = []
image_counts = []

for class_folder in os.listdir(root_directory):
    class_path = os.path.join(root_directory, class_folder)
    if os.path.isdir(class_path):
        class_names.append(class_folder)
        image_counts.append(len(os.listdir(class_path)))

# ------------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.bar(class_names, image_counts)
plt.xlabel('Plant Class')
plt.ylabel('Number of Images')
plt.title('Number of Images in Each Plant Class')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------

output_directory_test_train = '../Dataset/'
os.makedirs(os.path.join(output_directory_test_train, 'Train'), exist_ok=True)
os.makedirs(os.path.join(output_directory_test_train, 'Test'), exist_ok=True)
os.listdir(output_directory_test_train)

train_ratio = 0.8

for class_folder in os.listdir(input_data_dir):
    class_path = os.path.join(input_data_dir, class_folder)
    if os.path.isdir(class_path):
        image_files = os.listdir(class_path)
        num_images = len(image_files)
        num_train = int(train_ratio * num_images)

        train_class_dir = os.path.join(output_directory_test_train, 'Train', class_folder)
        test_class_dir = os.path.join(output_directory_test_train, 'Test', class_folder)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for i, image_file in enumerate(tqdm(image_files, desc=f'Copying {class_folder}')):
            source_path = os.path.join(class_path, image_file)
            if i < num_train:
                destination_path = os.path.join(train_class_dir, image_file)
            else:
                destination_path = os.path.join(test_class_dir, image_file)
            shutil.copyfile(source_path, destination_path)
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------
    
dataset_directory = '../Dataset/Plants/'
image_class_name = [image_class for image_class in os.listdir(dataset_directory)]
image_class_name_label = {image_class:index for index, image_class in enumerate(image_class_name)}
total_class = len(image_class_name)
IMAGE_SIZE = (160,160)

def pre_process(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    return image

def load_data():
    
    datasets = ['../Dataset/Train/', '../Dataset/Test/']
    output = []
    
    for dataset in datasets:
        
        images = []
        labels = []
        
        print(f"Loading {dataset}")
        
        for folder in os.listdir(dataset):
            label = image_class_name_label[folder]
            
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = pre_process(img_path) 
                
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()

# ------------------------------------------------------------------------------------------------------------------------------------------------------

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
number_train = train_labels.shape[0]
number_test = test_labels.shape[0]

print (f"Number of training examples: {number_train}")
print (f"Number of testing examples: {number_test}")
print (f"Each image is of size: {IMAGE_SIZE}")

train_images = train_images / 255.0 
test_images = test_images / 255.0

def display_examples(image_class_name, images, labels):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(image_class_name[labels[i]])
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160,160,3))
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  
    layers.Dense(256, activation='relu'),  
    layers.Dropout(0.5),
    layers.Dense(40, activation='softmax')
])

def lr_schedule(epoch):
    initial_lr = 0.001  
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.1  
    else:
        return initial_lr * 0.01  
    
lr_scheduler = LearningRateScheduler(lr_schedule)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  
              metrics=['accuracy'])

train_labels_encoded = to_categorical(train_labels, num_classes=40)
test_labels_encoded = to_categorical(test_labels, num_classes=40)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

history = model.fit(train_images, train_labels, epochs=50, batch_size=8, validation_split=0.2, callbacks=[early_stopping, lr_scheduler]) 

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')

model.save('../Model_Weights/Herbal-AI-Classification.h5')
model.save_weights('../Model_Weights/Herbal-AI-Classification-Weights.h5')

# ------------------------------------------------------------------------------------------------------------------------------------------------------