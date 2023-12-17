import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()

model = make_or_restore_model()
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_dataset = valid_data_generator.flow_from_directory('Cat_Dog_Dataset/test_set',
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary')
# Load the weights of the trained model for the specified epoch
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint_file = f"{checkpoint_dir}/cp-{epoch:04d}.ckpt"
# model.load_weights(checkpoint_file)

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
