import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# class my_model(tf.keras.Model):
#     def __init__(self, learning_rate):
#         super(my_model, self).__init__()
#         self.conv1d_1 = layers.Conv1D(1, 3, strides=2, activation='relu')
#         self.conv1d_2 = layers.Conv1D(1, 3, strides=2, activation='relu')
#         self.d1 = layers.Dense(16, activation='relu')
#         self.d2 = layers.Dense(16, activation='relu')
#         self.d3 = layers.Dense(3, activation=None)
#         self.optimizer = optimizers.Adam(learning_rate=learning_rate)
#
#     def call(self, x):
#         x = tf.expand_dims(x, axis=2)
#         lazer = x[:,0:15,:]
#         others = x[:,15:,:]
#         lazer = self.conv1d_1(lazer)
#         lazer = self.conv1d_2(lazer)
#         lazer = tf.cast(lazer, dtype=tf.float32)
#         others = tf.cast(others, dtype=tf.float32)
#         total = tf.concat([lazer, others], axis=1)
#         print('total shape1 : ', total.shape, len(total.shape))
#         total = tf.squeeze(total)
#         total = self.d1(total)
#         total = self.d2(total)
#         total = self.d3(total)
#         return total
#
#     def fit(self, x, y):
#         with tf.GradientTape() as tape:
#             pre_y = self.call(x)
#             loss = tf.keras.losses.mean_squared_error(pre_y, y)
#             # print('loss shape : ', loss.shape)
#             loss_sum = tf.reduce_mean(loss)
#             # print('loss_sum shape : ', loss_sum.shape)
#         gradients = tape.gradient(loss_sum, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#         return loss_sum

class my_model(tf.keras.Model):
    def __init__(self, lazer_num, output_size, learning_rate):
        super(my_model, self).__init__()
        self.conv1d_1 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_2 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_3 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.conv1d_4 = layers.Conv1D(1, 3, strides=1, activation='relu')
        self.d1 = layers.Dense(16, activation='relu')
        self.d2 = layers.Dense(16, activation='relu')
        self.d3 = layers.Dense(output_size, activation=None)
        self.lazer_num = lazer_num
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.conv1d_3 = layers.Conv1D(1, 3, strides=2, activation='relu')
        # self.conv1d_4 = layers.Conv1D(16, 3, activation='relu')
        # self.conv1d_5 = layers.Conv1D(32, 3, activation='relu')
        # self.conv1d_6 = layers.Conv1D(64, 3, activation='relu')

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        # lazer = x[:,0:self.lazer_num,:]
        lazer = x[:, 0:, :]
        # others = x[:,self.lazer_num:,:]
        # 处理激光数据
        # lazer = self.conv1d_1(lazer)
        # lazer = self.conv1d_2(lazer)
        # lazer = self.conv1d_3(lazer)
        # lazer = self.conv1d_4(lazer)
        # 数据类型转换
        lazer = tf.cast(lazer, dtype=tf.float32)
        # others = tf.cast(others, dtype=tf.float32)
        # 数据类型结合继续计算
        # total = tf.concat([lazer, others], axis=1)
        # print(total.shape)
        if len(lazer.shape) == 3:
            total = tf.squeeze(lazer, axis=2)
        total = self.d1(total)
        total = self.d2(total)
        total = self.d3(total)
        return total

    def fit(self, x, y):
        with tf.GradientTape() as tape:
            pre_y = self.call(x)
            loss = tf.keras.losses.mean_squared_error(pre_y, y)
            # print('loss shape : ', loss.shape)
            loss_sum = tf.reduce_mean(loss)
            # print('loss_sum shape : ', loss_sum.shape)
        gradients = tape.gradient(loss_sum, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_sum

model1 = my_model(15, 3, 0.001)
model2 = my_model(15, 3, 0.001)

# model1.set_weights(model2.get_weights())


x = np.random.normal(size=(32, 19))
# print(x)
print(x.shape)

pre_y = model1.call(x)

print(pre_y.shape)

y = np.random.normal(size=(32,3))
l = []
for i in range(500):
    l.append(model1.fit(x, y))
plt.plot(l)
plt.show()


