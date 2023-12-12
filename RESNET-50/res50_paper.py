import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
from tensorflow.python.client import device_lib
gpu_available = any(device.device_type == 'GPU' for device in device_lib.list_local_devices())

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.compat.v1.disable_eager_execution()
# tf.test.is_gpu_available()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# GPU 사용 여부 확인
gpu_available = any(device.device_type == 'GPU' for device in device_lib.list_local_devices())
print("GPU Available:", gpu_available)

'''
< ResNet Architecture>
- ResNet "50"-layer
- 5_x Layer (1,3,4,6,3)
- skip connection
- Sequential model X
- Batch Normalization right after each convolution and before activation
'''

def ResNet(x):
    # input = 224 x 224 x 3

    # Conv1 -> 1
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', input_shape=(224, 224, 3))(x)  # 112x112x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D((3, 3), 2, padding='same')(x)  # 56x56x64
    shortcut = x

    # Conv2_x -> 3
    for i in range(3) :
        if i==0 :
            x = layers.Conv2D(64, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            # In case of i = 0 (for Dimension Identity)
            # shortcut should enter as input with x
            shortcut = layers.Conv2D(256, (1, 1), strides=1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            shortcut = layers.Activation('relu')(shortcut)

            x = layers.Add()([x, shortcut])
            shortcut = x    # 56x56x256

        else :
            x = layers.Conv2D(64, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Add()([x, shortcut])
            shortcut = x    # 56x56x256

    # Conv3_x -> 4
    for i in range(4) :
        if i==0 :
            x = layers.Conv2D(128, (1, 1), strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(512, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            # In case of i = 0 (for Dimension Identity)
            # shortcut should enter as input with x, 112x112x64 -> 112x112x256
            shortcut = layers.Conv2D(512, (1, 1), strides=2, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            shortcut = layers.Activation('relu')(shortcut)

            x = layers.Add()([x, shortcut])
            shortcut = x    # 28x28x512

        else :
            x = layers.Conv2D(128, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(512, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Add()([x, shortcut])
            shortcut = x    # 28x28x512

    # Conv4_x -> 6
    for i in range(6) :
        if i==0 :
            x = layers.Conv2D(256, (1, 1), strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(1024, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            # In case of i = 0 (for Dimension Identity)
            # shortcut should enter as input with x, 112x112x64 -> 112x112x256
            shortcut = layers.Conv2D(1024, (1, 1), strides=2, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            shortcut = layers.Activation('relu')(shortcut)

            x = layers.Add()([x, shortcut])
            shortcut = x    # 14x14x1024

        else :
            x = layers.Conv2D(256, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(1024, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Add()([x, shortcut])
            shortcut = x    # 14x14x1024

    # Conv5_x -> 3
    for i in range(3) :
        if i==0 :
            x = layers.Conv2D(512, (1, 1), strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(512, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(2048, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            # In case of i = 0 (for Dimension Identity)
            # shortcut should enter as input with x, 112x112x64 -> 112x112x256
            shortcut = layers.Conv2D(2048, (1, 1), strides=2, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            shortcut = layers.Activation('relu')(shortcut)

            x = layers.Add()([x, shortcut])
            shortcut = x    # 7x7x2048

        else :
            x = layers.Conv2D(512, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(512, (3, 3), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(2048, (1, 1), strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Add()([x, shortcut])
            shortcut = x    # 7x7x2048

    # 2048 (same with AdaptiveAvgPool in Pytorch)
    x = layers.GlobalAveragePooling2D()(x)
    # classes = 2
    x = layers.Dense(2, activation='softmax')(x)

    return x


# Dataset (Kaggle Cat and Dog Dataset)
dataset_path = os.path.join('./Cat_Dog_Dataset')
train_dataset_path = dataset_path + '/train_set'
train_data_generator = ImageDataGenerator(rescale=1. / 255)
train_dataset = train_data_generator.flow_from_directory(train_dataset_path,
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')

valid_dataset_path = dataset_path + '/test_set'
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_data_generator.flow_from_directory(valid_dataset_path,
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')



input_shape = layers.Input(shape=(224, 224, 3), dtype='float32', name='input')
# Train
model = tf.keras.Model(input_shape, ResNet(input_shape))
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
#                   loss='categorical_crossentropy',
#                   metrics=['acc'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
train = model.fit_generator(train_dataset, epochs=5, validation_data=valid_dataset)
