import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

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


# def cosine_decay_with_warmup(global_step,
#                              learning_rate_base,
#                              total_steps,
#                              warmup_learning_rate=0.0,
#                              warmup_steps=0,
#                              hold_base_rate_steps=0):
    
#     if total_steps < warmup_steps:
#         raise ValueError('total_steps must be larger or equal to '
#                          'warmup_steps.')
#     learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
#         np.pi *
#         (global_step - warmup_steps - hold_base_rate_steps
#          ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
#     if hold_base_rate_steps > 0:
#         learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
#                                  learning_rate, learning_rate_base)
#     if warmup_steps > 0:
#         if learning_rate_base < warmup_learning_rate:
#             raise ValueError('learning_rate_base must be larger or equal to '
#                              'warmup_learning_rate.')
#         slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
#         warmup_rate = slope * global_step + warmup_learning_rate
#         learning_rate = np.where(global_step < warmup_steps, warmup_rate,
#                                  learning_rate)
#     return np.where(global_step > total_steps, 0.0, learning_rate)


# Warmup Learning Rate Scheduler
class WarmUpLearningRateScheduler(Callback):
    def __init__(self, warmup_batches, init_lr, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning rate to %s.' % (self.batch_count + 1, lr))


# class WarmUpCosineDecayScheduler(Callback):

#     def __init__(self,
#                  learning_rate_base,
#                  total_steps,
#                  global_step_init=0,
#                  warmup_learning_rate=0.0,
#                  warmup_steps=0,
#                  hold_base_rate_steps=0,
#                  verbose=0):


#         super(WarmUpCosineDecayScheduler, self).__init__()
#         self.learning_rate_base = learning_rate_base
#         self.total_steps = total_steps
#         self.global_step = global_step_init
#         self.warmup_learning_rate = warmup_learning_rate
#         self.warmup_steps = warmup_steps
#         self.hold_base_rate_steps = hold_base_rate_steps
#         self.verbose = verbose
#         self.learning_rates = []

#     def on_batch_end(self, batch, logs=None):
#         self.global_step = self.global_step + 1
#         lr = K.get_value(self.model.optimizer.lr)
#         self.learning_rates.append(lr)

#     def on_batch_begin(self, batch, logs=None):
#         lr = cosine_decay_with_warmup(global_step=self.global_step,
#                                       learning_rate_base=self.learning_rate_base,
#                                       total_steps=self.total_steps,
#                                       warmup_learning_rate=self.warmup_learning_rate,
#                                       warmup_steps=self.warmup_steps,
#                                       hold_base_rate_steps=self.hold_base_rate_steps)
#         K.set_value(self.model.optimizer.lr, lr)
#         if self.verbose > 0:
#             print('\nBatch %05d: setting learning '
#                   'rate to %s.' % (self.global_step + 1, lr))

    

# 데이터 증강 및 전처리 옵션
train_data_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'  # 따옴표 수정
)

valid_data_generator = ImageDataGenerator(rescale=1. / 255)

# Parameter
batch_size = 128
epochs = 15
directory_path = "../../Dataset"

# 학습 데이터셋 및 검증 데이터셋 생성
train_dataset = train_data_generator.flow_from_directory(
    f'{directory_path}/train_data',
    shuffle=True,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

valid_dataset = valid_data_generator.flow_from_directory(
    f'{directory_path}/test_data',
    shuffle=True,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)


# Function to adjust prediction threshold
def adjust_threshold(y_pred, threshold=0.5):
    return (y_pred > threshold).astype(int)

# Model parameters
sample_count = 120
warmup_epoch = 5
learning_rate_base = 0.001

# Compute the number of warmup batches.
warmup_batches = warmup_epoch * sample_count / batch_size

# Create the Learning rate scheduler.
warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=0.001)



# # Create the Learning rate scheduler.
# total_steps = int(epochs * sample_count / batch_size)
# warmup_steps = int(warmup_epoch * sample_count / batch_size)
# warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
#                                         total_steps=total_steps,
#                                         warmup_learning_rate=0.0,
#                                         warmup_steps=warmup_steps,
#                                         hold_base_rate_steps=0)

# Create the model.
input_shape = layers.Input(shape=(224, 224, 3), dtype='float32', name='input')
output = DenseNet(input_shape)

#Flatten 추가
x = Flatten()(output)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_shape, outputs=x)

# Compile the model.
optimizer = Adam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy", f1_m])


checkpoint_path = "ckpts_2/cp-{epoch:04d}.ckpt"
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


# Train the model using fit_generator with class weights
model.fit_generator(train_dataset, epochs=epochs, steps_per_epoch=sample_count // batch_size,
                    verbose=1, callbacks=[warm_up_lr, cp_callback], validation_data=valid_dataset)


# Evaluate the model on the validation set
val_loss, val_accuracy, val_f1 = model.evaluate(valid_dataset)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}")

# Make predictions on the validation set
val_predictions = model.predict(valid_dataset)
# Adjust the threshold for binary classification (if needed)
val_predictions_adjusted = adjust_threshold(val_predictions, threshold=0.5)

# 파일명 설정
file_name = "test_metrics_opt.txt"

# 결과를 텍스트 파일에 저장
with open(file_name, "w") as file:
    file.write(f"Test Loss: {val_loss}\n")
    file.write(f"Test Accuracy: {val_accuracy}\n")
    file.write(f"Test F1 Score: {val_f1}\n")

print(f"Test metrics have been saved to {file_name}.")
