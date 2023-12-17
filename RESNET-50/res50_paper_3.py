import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam

file_path = './archive'

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

from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
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
train_data_generator = ImageDataGenerator(rescale=1. / 255)
train_dataset = train_data_generator.flow_from_directory(f'{file_path}/train_data',
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary')

valid_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_data_generator.flow_from_directory(f'{file_path}/test_data',
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary')

input_shape = layers.Input(shape=(224, 224, 3), dtype='float32', name='input')
# Train
model = tf.keras.Model(input_shape, ResNet(input_shape))
#########################################################################################################
#하이퍼파라미터 튜닝으로 아래 내용 주석처리
#########################################################################################################
# # Adam 옵티마이저 생성
# optimizer = Adam(learning_rate=0.001)  # 여기서 learning_rate를 설정

# # 모델 컴파일
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy", f1_m])

# model.summary()
# # Include the epoch in the file name (uses `str.format`)
# checkpoint_path = "ckpts/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# batch_size = 211

# # Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     save_freq=batch_size)


# # Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))

# train = model.fit(train_dataset, epochs=5, callbacks=[cp_callback], validation_data=valid_dataset)

#########################################################################################################
#하이퍼파라미터 튜닝
#########################################################################################################
# Define the objective function for Optuna optimization
import optuna
from tensorflow.keras.optimizers import Adam, Nadam, Adamax

# Define the objective function for Optuna optimization
def objective(trial):
    # Define the hyperparameters to be tuned
    learning_rate = trial.suggest_discrete_uniform('learning_rate', 1e-5, 1e-2, 2)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'nadam'])

    # Choose the optimizer based on the optimizer_name
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamax':
        optimizer = Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy", f1_m])

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "ckpts/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    batch_size = 211

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=batch_size)

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # Train the model with the current hyperparameters
    train_history = model.fit(train_dataset, epochs=5, callbacks=[cp_callback], validation_data=valid_dataset)

    # Return the validation accuracy as the objective value to be maximized
    return train_history.history['val_accuracy'][-1]

# Run Optuna optimization
study = optuna.create_study(direction='maximize')  # We want to maximize validation accuracy
study.optimize(objective, n_trials=4)  # You can adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)