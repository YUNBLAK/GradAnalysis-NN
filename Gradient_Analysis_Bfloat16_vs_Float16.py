import random
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

arr1 = []
arr2 = []
arr3 = []
arr4 = []
arr5 = []

def seed_everything(seed: int = 51):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

def custom_relu(x):
    return tf.where(x > 0, x, 0)

fp = "bfloat16"
seed_everything(52)
tf.keras.mixed_precision.set_global_policy(fp)

# MNIST 데이터 로드
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 정규화

fp = "float32"

train_images = (train_images.reshape(60000, 784) / 255).astype(fp)
test_images = (test_images.reshape(10000, 784) / 255).astype(fp)

inputs = keras.Input(shape=(784,), name='digits')

if tf.config.list_physical_devices('GPU'):
    num_units = 2048
else:
    num_units = 64

dense1 = layers.Dense(num_units, name='dense_1', activation = custom_relu, dtype = fp)
x = dense1(inputs)
dense2 = layers.Dense(num_units, name='dense_2', activation = custom_relu, dtype = fp)
x = dense2(x)
dense3 = layers.Dense(num_units, name='dense_3', activation = custom_relu, dtype = fp)
x = dense3(x)
# DNN 모델 정의

x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype=fp, name='predictions')(x)
# 모델 컴파일
model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate = 0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 그래디언트 데이터 저장을 위한 리스트
stored_data = []


from sklearn.decomposition import PCA
class GradientVisualization(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            logits = model(train_images, training=True)
            loss_value = tf.keras.losses.sparse_categorical_crossentropy(train_labels, logits, from_logits=True)
        grads = tape.gradient(loss_value, model.trainable_weights)

        flattened_gradients = []
        for g in grads:
            flattened_gradients.append(tf.reshape(g, [-1]).numpy())
        flattened_gradients = np.concatenate(flattened_gradients, axis=0)

        # Count the number of zeros in the flattened_gradients
        # Count the number of zeros in the flattened_gradients
        print()
        print()
        zero_count = np.sum(flattened_gradients == 0)
        print("The number of 0:", zero_count)
        
        flattened_gradients = tf.cast(flattened_gradients, tf.float32)
        # Compute the standard deviation of flattened_gradients using numpy
        standard_deviation = np.std(flattened_gradients, ddof=1)
        print("Standard Deviation:", standard_deviation)

        # Compute the variance of flattened_gradients using numpy
        variance = np.var(flattened_gradients, ddof=1)
        print("Variance:", variance)

        # Compute the maximum and minimum values of flattened_gradients using numpy
        max_value = np.max(flattened_gradients)
        min_value = np.min(flattened_gradients)
        print("Max Value:", max_value)
        print("Min Value:", min_value)
        print()
        print()
        global arr1, arr2, arr3, arr4, arr5
        arr1.append(zero_count)
        arr2.append(standard_deviation)
        arr3.append(variance)
        arr4.append(max_value)
        arr5.append(min_value)

        #self.visualize_gradients(epoch, flattened_gradients)

    def visualize_gradients(self, epoch, gradients):
        plt.figure(figsize=(8, 6))
        plt.hist(gradients, bins=100, color='blue', alpha=0.7)
        plt.title(f"Gradient Distribution for Epoch: {epoch}")
        plt.xlabel('Gradient Magnitude')
        plt.ylabel('Count')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

# 모델 학습
history = model.fit(train_images,
          train_labels,
          epochs=100,
          batch_size=128,
          validation_data=(test_images, test_labels),
          callbacks=[GradientVisualization()]
          )

acc = history.history['val_accuracy']
tacc = history.history['accuracy']
dict = {
    "zero": arr1,
    "std" : arr2,
    "var" : arr3,
    "max" : arr4,
    "min" : arr5,
    "acc" : acc,
    "tacc": tacc
}

df = pd.DataFrame(dict)
df.to_csv("test.csv", index = False)
