import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.test.is_gpu_available()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from keras import backend as K

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
< DenseNet Architecture>
- Dense Connectivity pattern
- Dense-121, Dense-169, Dense-201, Dense-264
- Implement Dense-121 (6, 12, 24, 16)
'''

def DenseNet(x):
    # input = 224 x 224 x 3
    k = 32  # Grow Rate
    compression = 0.5   # compression factor

    # 1. Convolution
    x = layers.Conv2D(k * 2, (7, 7), strides=2, padding='same', input_shape=(224, 224, 3))(x)    # 112x112x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. Pooling
    x = layers.MaxPool2D((3, 3), 2, padding='same')(x)  # 56x56x64

    # 3. Dense Block (1)
    for i in range(6) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)    # 56x56x128
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)  # 56x56x32
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])  # 96 -> 128 -> 160 -> 192 -> 224 -> 256

    # 4. Transition Layer (1)
    current_shape = int(x.shape[-1]) # 56x56x256
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 28x28

    # 5. Dense Block (2)
    for i in range(12) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 6. Transition Layer (2)
    current_shape = int(x.shape[-1])
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 14x14

    # 7. Dense Block (3)
    for i in range(24) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 8. Transition Layer (3)
    current_shape = int(x.shape[-1])
    x = layers.Conv2D(int(current_shape * compression), (1, 1), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)   # 7x7

    # 9. Dense Block (4)
    for i in range(16) :
        x_l = layers.Conv2D(k * 4, (1, 1), strides=1, padding='same')(x)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x_l = layers.Conv2D(k, (3, 3), strides=1, padding='same')(x_l)
        x_l = layers.BatchNormalization()(x_l)
        x_l = layers.Activation('relu')(x_l)

        x = layers.Concatenate()([x, x_l])

    # 10. Classification Layer
    x = layers.GlobalAveragePooling2D()(x)
    # classes = 2 (softmax)
    x = layers.Dense(1, activation='sigmoid')(x)

    return x

# Parameter
batch_size = 256
epoch = 5
learning_rate = 0.001
directory_path = "../../Dataset"

# Dataset (Kaggle Cat and Dog Dataset)
train_data_generator = ImageDataGenerator(rescale=1. / 255)
train_dataset = train_data_generator.flow_from_directory(f'{directory_path}/train_data',
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size= batch_size,
                                                         class_mode='binary')

valid_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_data_generator.flow_from_directory(f'{directory_path}/test_data',
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size= batch_size,
                                                         class_mode='binary')

input_shape = layers.Input(shape=(224, 224, 3), dtype='float32', name='input')
# Train
model = tf.keras.Model(input_shape, DenseNet(input_shape))
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
#                   loss='categorical_crossentropy',
#                   metrics=['acc'])

# Adam 옵티마이저 생성
optimizer = Adam(learning_rate=learning_rate)  # 여기서 learning_rate를 설정

# 모델 컴파일
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy", f1_m])

model.summary()
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "ckpts_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size_cp = 27000 / batch_size + 1

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=batch_size_cp)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

train = model.fit(train_dataset, epochs=epoch, callbacks=[cp_callback], validation_data=valid_dataset)

# Evaluate the model on the validation set
val_loss, val_accuracy, val_f1 = model.evaluate(valid_dataset)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}")


def load_and_evaluate_model(checkpoint_dir, epoch, valid_dataset):
    # Load the weights of the trained model for the specified epoch
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_file = f"{checkpoint_dir}/cp-{epoch:04d}.ckpt"
    model.load_weights(checkpoint_file)

    # Evaluate the model on the validation dataset
    val_loss, val_accuracy, val_f1 = model.evaluate(valid_dataset)

    # Extracting predictions and true labels
    y_pred = model.predict(valid_dataset)
    y_true = valid_dataset.labels

    # Convert probabilities to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate precision, recall, and F1 score
    epsilon_value = tf.keras.backend.epsilon()
    true_positives = np.sum((y_true == 1) & (y_pred_classes == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_classes == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_classes == 0))

    precision = true_positives / (true_positives + false_positives + epsilon_value)
    recall = true_positives / (true_positives + false_negatives + epsilon_value)
    f1score = 2 * (precision * recall) / (precision + recall + epsilon_value)

    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation F1 Score_1: {f1score}")
    print(f"Validation F1 Score_2: {val_f1}")

    # 파일명 설정
    file_name = "test_metrics.txt"

    # 결과를 텍스트 파일에 저장
    with open(file_name, "w") as file:
        file.write(f"Test Loss: {val_loss}\n")
        file.write(f"Test Accuracy: {val_accuracy}\n")
        file.write(f"Validation F1 Score_1: {f1score}")
        file.write(f"Test F1 Score: {val_f1}\n")

    print(f"Test metrics have been saved to {file_name}.")

    return model

checkpoint_dir = "ckpts"
epoch_to_load = 3

# load_and_evaluate_model(checkpoint_dir, epoch_to_load, valid_dataset)